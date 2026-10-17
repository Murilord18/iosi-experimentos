"""
validador_19h.py
================
Fase E - Projeto IOSI. Conferencia do dia as 19h (BRT).

Le a previsao do dia (data/previsoes/AAAA-MM-DD.json, a primeira do dia, gerada as 9h15)
e o candle diario real do BOVA11 no snapshot das 19h (data/snapshots/AAAA-MM-DD_19h.csv),
e acrescenta uma linha em data/historico_previsoes.csv com:
    - previsoes (gap com EWZ das 9h e das 9h15, gap usado no card, abertura, amplitude,
      faixa provavel, prob. de lateral);
    - valores reais (open, high, low, close, gap, amplitude, |close-open|/open, lateral);
    - erro de cada alvo e se a maxima/minima ficaram dentro da faixa;
    - baselines do dia (har_5d, gap = 0, taxa de lateral dos ultimos 252 pregoes) e seus erros.

O gap real usa o mesmo fechamento anterior da previsao (campo fech_anterior do JSON).
Se o dia ja estiver no historico, nada e gravado (o historico nunca e reescrito).
Sem previsao ou sem pregao, grava uma linha com o status correspondente.

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
from predictor_9h15 import pastas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLUNAS = [
    "data", "status", "validado_em", "simulacao", "previsao_gerada_em", "fonte_gap_card", "fonte_candle",
    "fech_anterior", "gap_prev_9h", "gap_prev_9h15", "gap_prev", "abertura_prev", "amplitude_prev", "faixa_inf", "faixa_sup", "prob_lateral",
    "open", "high", "low", "close", "gap_real", "amplitude_real", "abs_intraday_real", "lateral_real",
    "erro_gap_9h", "erro_gap_9h15", "erro_gap", "erro_abertura_rs", "erro_amplitude", "brier_lateral", "max_dentro_faixa", "min_dentro_faixa",
    "amplitude_har_5d", "erro_amplitude_har_5d", "erro_gap_zero", "taxa_lateral_252d", "brier_lateral_taxa_252d",
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
            "fonte_candle": f"{arq.name} ({r['fonte']})"}


def validar(t: pd.Timestamp, dir_previsoes: Path, dir_snapshots: Path) -> dict:
    base = {"data": str(t.date()), "validado_em": pd.Timestamp.now(tz=ds.TZ).strftime("%Y-%m-%d %H:%M:%S%z")}
    arq_prev = dir_previsoes / f"{t:%Y-%m-%d}.json"
    if not arq_prev.exists():
        return {**base, "status": "sem_previsao"}
    p = json.loads(arq_prev.read_text(encoding="utf-8"))
    if p["status"] != "ok":
        return {**base, "status": p["status"]}
    real = candle_real(t, dir_snapshots)
    if not real:
        return {**base, "status": "sem_candle_real"}

    gap_real = real["open"] / p["fech_anterior"] - 1
    amp_real = (real["high"] - real["low"]) / real["open"]
    abs_intra = abs(real["close"] / real["open"] - 1)
    lateral = int(abs_intra < ds.LIMIAR_LATERAL)
    b = p["baselines"]
    return {
        **base, "status": "ok", "simulacao": p.get("simulacao", False),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="pregao a validar (AAAA-MM-DD); padrao: hoje")
    parser.add_argument("--simulacao", action="store_true", help="usa data/simulacao/")
    parser.add_argument("--snapshots", type=Path, default=ds.DIR_SNAPSHOTS)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    t = pd.Timestamp(args.data) if args.data else ds.HOJE
    p = pastas(args.simulacao)
    linha = validar(t, p["previsoes"], args.snapshots)

    hist = p["historico"]
    if hist.exists() and str(t.date()) in pd.read_csv(hist, usecols=["data"])["data"].astype(str).values:
        logger.warning("%s ja esta no historico (%s); nada gravado", t.date(), hist)
        return 0
    hist.parent.mkdir(parents=True, exist_ok=True)
    novo = pd.DataFrame([linha]).reindex(columns=COLUNAS)
    novo.to_csv(hist, mode="a", header=not hist.exists(), index=False, float_format="%.10f")
    logger.info("Linha gravada em %s (status: %s)", hist, linha["status"])
    print(pd.Series(linha).to_string())
    # dia sem previsao fica registrado no historico, mas nao e erro de execucao
    return 0 if linha["status"] in ("ok", "sem_pregao", "sem_previsao") else 1


if __name__ == "__main__":
    sys.exit(main())
