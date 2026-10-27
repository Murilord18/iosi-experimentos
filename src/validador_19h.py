"""
validador_19h.py
================
Fase E - Projeto IOSI. Conferencia do dia as 19h (BRT).

Le a previsao do dia, a primeira do dia: data/previsoes/AAAA-MM-DD_1030.json (10h30, a
atual) ou, se nao existir, AAAA-MM-DD.json (9h15, usada ate 16/10/2026). Le tambem o candle
diario real do BOVA11 e os candles de 5m do pregao no snapshot das 19h
(data/snapshots/AAAA-MM-DD_19h.csv), e acrescenta uma linha em data/historico_previsoes.csv com:
    - previsoes (amplitude, faixa provavel, prob. de lateral; na das 9h15 tambem o gap e a
      abertura esperada; na das 10h30, a abertura real e o preco as 10h30);
    - valores reais (open, high, low, close, gap, amplitude, |close-open|/open, lateral);
    - erro de cada alvo e se a maxima/minima do dia ficaram dentro da faixa;
    - resto do dia (previsao das 10h30), com os candles de 5m que comecam depois do preco
      das 10h30: ret_resto = close / preco_1030 - 1, amp_resto = (maxima - minima depois de
      10h30) / preco_1030 e lateral_resto = |ret_resto| < 1%;
    - baselines do dia (har_5d, gap = 0, taxa de lateral dos ultimos 252 pregoes) e seus erros.

O gap real usa o mesmo fechamento anterior da previsao (campo fech_anterior do JSON).
O historico nunca tem valores reescritos; se o cabecalho for de uma versao anterior
(sem as colunas das 10h30), o arquivo e regravado com as colunas novas vazias nas linhas antigas.
Se o dia ja estiver no historico, nada e gravado (o historico nunca e reescrito).
Sem previsao ou sem pregao, grava uma linha com o status correspondente.
A linha do dia (como esta no CSV) tambem vai para a tabela previsoes do Supabase, e a
execucao e registrada em execucoes (storage_supabase.py).

Uso:
    python src/validador_19h.py
    python src/validador_19h.py --data 2026-10-16 --simulacao [--snapshots <pasta>]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

import dataset_9h15 as ds
import leitura_1030 as l1030
import storage_supabase as sb
from predictor_9h15 import pastas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLUNAS = [
    "data", "status", "validado_em", "simulacao", "previsao_gerada_em", "fonte_gap_card", "fonte_candle",
    "fech_anterior", "gap_prev_9h", "gap_prev_9h15", "gap_prev", "abertura_prev", "amplitude_prev", "faixa_inf", "faixa_sup", "prob_lateral",
    "open", "high", "low", "close", "gap_real", "amplitude_real", "abs_intraday_real", "lateral_real",
    "erro_gap_9h", "erro_gap_9h15", "erro_gap", "erro_abertura_rs", "erro_amplitude", "brier_lateral", "max_dentro_faixa", "min_dentro_faixa",
    "amplitude_har_5d", "erro_amplitude_har_5d", "erro_gap_zero", "taxa_lateral_252d", "brier_lateral_taxa_252d",
    "previsao_hora", "abertura_real_1030", "preco_1030", "preco_1030_fim", "high_ate_1030", "low_ate_1030",
    "faixa_cobertura_card", "ret_resto", "abs_ret_resto", "amp_resto", "lateral_resto", "high_resto", "low_resto",
    "n_candles_resto", "fonte_resto",
]


def candle_real(t: pd.Timestamp, dir_snapshots: Path) -> dict:
    arq = dir_snapshots / f"{t:%Y-%m-%d}_19h.csv"
    if not arq.exists():
        return {}
    s = pd.read_csv(arq)
    d = s[(s["ativo"] == "BOVA11") & (s["tipo"] == "1d")]
    if d.empty:
        return {}
    r = d.iloc[0]
    return {"open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
            "fonte_candle": f"{arq.name} ({r['fonte']})", "arquivo": arq}


def ler_previsao(t: pd.Timestamp, dir_previsoes: Path) -> dict | None:
    """Primeira previsao do dia: a das 10h30 ou, sem ela, a das 9h15."""
    for nome in (f"{t:%Y-%m-%d}_1030.json", f"{t:%Y-%m-%d}.json"):
        arq = dir_previsoes / nome
        if arq.exists():
            return json.loads(arq.read_text(encoding="utf-8"))
    return None


def resto_do_dia(p: dict, real: dict) -> dict:
    """Metricas depois do preco das 10h30, com os candles de 5m do pregao gravados as 19h."""
    preco, ref = p["preco_1030"], pd.Timestamp(p["preco_1030_fim"])
    candles, _ = l1030.ler_snapshot(real["arquivo"])
    c = candles.get("BOVA11")
    depois = c[c["inicio"] >= ref] if c is not None else pd.DataFrame()
    ret = real["close"] / preco - 1
    linha = {"ret_resto": ret, "abs_ret_resto": abs(ret), "lateral_resto": int(abs(ret) < ds.LIMIAR_LATERAL),
             "n_candles_resto": len(depois), "fonte_resto": f"{real['arquivo'].name} (5m, inicio >= {ref:%H:%M})"}
    if len(depois):
        alta, baixa = float(depois["high"].max()), float(depois["low"].min())
        linha.update({"high_resto": alta, "low_resto": baixa, "amp_resto": (alta - baixa) / preco})
    return linha


def validar(t: pd.Timestamp, dir_previsoes: Path, dir_snapshots: Path) -> dict:
    base = {"data": str(t.date()), "validado_em": pd.Timestamp.now(tz=ds.TZ).strftime("%Y-%m-%d %H:%M:%S%z")}
    p = ler_previsao(t, dir_previsoes)
    if p is None:
        return {**base, "status": "sem_previsao"}
    hora = p.get("previsao_hora", "9h15")
    if p["status"] != "ok":
        return {**base, "status": p["status"], "previsao_hora": hora}
    real = candle_real(t, dir_snapshots)
    if not real:
        return {**base, "status": "sem_candle_real", "previsao_hora": hora}
    if hora == "10h30":
        return validar_1030(base, p, real)

    gap_real = real["open"] / p["fech_anterior"] - 1
    amp_real = (real["high"] - real["low"]) / real["open"]
    abs_intra = abs(real["close"] / real["open"] - 1)
    lateral = int(abs_intra < ds.LIMIAR_LATERAL)
    b = p["baselines"]
    return {
        **base, "status": "ok", "simulacao": p.get("simulacao", False), "previsao_hora": "9h15",
        "previsao_gerada_em": p["gerado_em"], "fonte_gap_card": p["fonte_gap_card"],
        "fonte_candle": real["fonte_candle"],
        "fech_anterior": p["fech_anterior"],
        # previsoes
        "gap_prev_9h": p["gap_prev_9h"], "gap_prev_9h15": p["gap_prev_9h15"],
        "gap_prev": p["gap_previsto"], "abertura_prev": p["abertura_esperada"],
        "amplitude_prev": p["amplitude_prevista"], "faixa_inf": p["faixa_inferior"],
        "faixa_sup": p["faixa_superior"], "prob_lateral": p["prob_lateral"],
        # reais
        "open": real["open"], "high": real["high"], "low": real["low"], "close": real["close"],
        "gap_real": gap_real, "amplitude_real": amp_real, "abs_intraday_real": abs_intra,
        "lateral_real": lateral,
        # erros do modelo
        "erro_gap_9h": p["gap_prev_9h"] - gap_real,
        "erro_gap_9h15": p["gap_prev_9h15"] - gap_real if p["gap_prev_9h15"] is not None else None,
        "erro_gap": p["gap_previsto"] - gap_real,
        "erro_abertura_rs": p["abertura_esperada"] - real["open"],
        "erro_amplitude": p["amplitude_prevista"] - amp_real,
        "brier_lateral": (p["prob_lateral"] - lateral) ** 2,
        "max_dentro_faixa": int(real["high"] <= p["faixa_superior"]),
        "min_dentro_faixa": int(real["low"] >= p["faixa_inferior"]),
        # baselines
        "amplitude_har_5d": b["amplitude_har_5d"], "erro_amplitude_har_5d": b["amplitude_har_5d"] - amp_real,
        "erro_gap_zero": -gap_real,
        "taxa_lateral_252d": b["taxa_lateral_252d"],
        "brier_lateral_taxa_252d": (b["taxa_lateral_252d"] - lateral) ** 2,
    }


def validar_1030(base: dict, p: dict, real: dict) -> dict:
    """Conferencia da previsao das 10h30: dia inteiro e resto do dia (depois de 10h30)."""
    gap_real = real["open"] / p["fech_anterior"] - 1
    amp_real = (real["high"] - real["low"]) / real["open"]
    abs_intra = abs(real["close"] / real["open"] - 1)
    lateral = int(abs_intra < ds.LIMIAR_LATERAL)
    b = p["baselines"]
    return {
        **base, "status": "ok", "simulacao": p.get("simulacao", False), "previsao_hora": "10h30",
        "previsao_gerada_em": p["gerado_em"], "fonte_candle": real["fonte_candle"],
        "fech_anterior": p["fech_anterior"],
        # previsao das 10h30 (a abertura ja e conhecida: nao ha gap previsto)
        "abertura_real_1030": p["abertura_real"], "preco_1030": p["preco_1030"],
        "preco_1030_fim": p["preco_1030_fim"], "high_ate_1030": p["high_ate_1030"],
        "low_ate_1030": p["low_ate_1030"],
        "amplitude_prev": p["amplitude_prevista"], "faixa_inf": p["faixa_inferior"],
        "faixa_sup": p["faixa_superior"], "faixa_cobertura_card": p["faixa_cobertura_card"],
        "prob_lateral": p["prob_lateral"],
        # reais do dia inteiro
        "open": real["open"], "high": real["high"], "low": real["low"], "close": real["close"],
        "gap_real": gap_real, "amplitude_real": amp_real, "abs_intraday_real": abs_intra,
        "lateral_real": lateral,
        # erros do modelo
        "erro_amplitude": p["amplitude_prevista"] - amp_real,
        "brier_lateral": (p["prob_lateral"] - lateral) ** 2,
        "max_dentro_faixa": int(real["high"] <= p["faixa_superior"]),
        "min_dentro_faixa": int(real["low"] >= p["faixa_inferior"]),
        # resto do dia
        **resto_do_dia(p, real),
        # baselines
        "amplitude_har_5d": b["amplitude_har_5d"], "erro_amplitude_har_5d": b["amplitude_har_5d"] - amp_real,
        "erro_gap_zero": -gap_real,
        "taxa_lateral_252d": b["taxa_lateral_252d"],
        "brier_lateral_taxa_252d": (b["taxa_lateral_252d"] - lateral) ** 2,
    }


def atualizar_cabecalho(hist: Path) -> None:
    """Historico de uma versao anterior: regrava com as colunas novas (valores antigos intactos)."""
    if not hist.exists():
        return
    cab = list(pd.read_csv(hist, nrows=0).columns)
    if cab == COLUNAS:
        return
    h = pd.read_csv(hist, dtype=str, keep_default_na=False)
    faltam = [c for c in COLUNAS if c not in cab]
    h.reindex(columns=COLUNAS + [c for c in cab if c not in COLUNAS], fill_value="").to_csv(hist, index=False)
    logger.info("Historico %s: cabecalho atualizado (+%d colunas: %s)", hist, len(faltam), ", ".join(faltam))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="pregao a validar (AAAA-MM-DD); padrao: hoje")
    parser.add_argument("--simulacao", action="store_true", help="usa data/simulacao/")
    parser.add_argument("--snapshots", type=Path, default=ds.DIR_SNAPSHOTS)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with sb.Execucao("noite", "validacao", ativo=not args.simulacao) as ex:
        codigo = conferir(args)
        if codigo:
            ex.falha("status sem_candle_real (sem snapshot das 19h com o candle do BOVA11)")
    return codigo


def enviar_supabase(hist: Path, t: pd.Timestamp) -> None:
    """Manda a linha do dia, como esta no historico, para a tabela previsoes."""
    try:
        h = pd.read_csv(hist)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supabase: historico nao lido (%s): %s", hist, exc)
        return
    sb.upsert_historico(h[h["data"].astype(str) == str(t.date())])


def conferir(args) -> int:
    t = pd.Timestamp(args.data) if args.data else ds.hoje()
    p = pastas(args.simulacao)
    linha = validar(t, p["previsoes"], args.snapshots)

    hist = p["historico"]
    atualizar_cabecalho(hist)
    if hist.exists() and str(t.date()) in pd.read_csv(hist, usecols=["data"])["data"].astype(str).values:
        logger.warning("%s ja esta no historico (%s); nada gravado", t.date(), hist)
        if not args.simulacao:
            enviar_supabase(hist, t)     # reenvio: se o Supabase falhou antes, completa agora
        return 0
    hist.parent.mkdir(parents=True, exist_ok=True)
    novo = pd.DataFrame([linha]).reindex(columns=COLUNAS)
    novo.to_csv(hist, mode="a", header=not hist.exists(), index=False, float_format="%.10f")
    logger.info("Linha gravada em %s (status: %s)", hist, linha["status"])
    if not args.simulacao:
        enviar_supabase(hist, t)
    print(pd.Series(linha).to_string())
    # dia sem previsao fica registrado no historico, mas nao e erro de execucao
    return 0 if linha["status"] in ("ok", "sem_pregao", "sem_previsao") else 1


if __name__ == "__main__":
    sys.exit(main())
