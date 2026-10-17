"""
predictor_9h15.py
=================
Fase E - Projeto IOSI. Previsao do dia as 9h15 (BRT), com a B3 fechada.

Monta a linha do dia com as mesmas funcoes do dataset_9h15.py, cortando os dados no
que era conhecido as 9h15 (truncar), e aplica os modelos de models/:
    gap.joblib        -> gap_prev_9h (EWZ do candle de 1h das 9h, igual ao treino) e
                         gap_prev_9h15 (EWZ do snapshot das 9h15, quando existir);
                         abertura esperada = fechamento anterior x (1 + gap), com o gap
                         das 9h15 (ou o das 9h, sem snapshot)
    amplitude.joblib  -> amplitude esperada (% e R$)
    faixa.json        -> faixa provavel [abertura x (1 - k_inf x amp), abertura x (1 + k_sup x amp)],
                         multiplicadores que contem maxima e minima em 80% dos dias com a
                         abertura real (calibrar_faixa.py); o card mostra a cobertura medida
                         com a abertura esperada (cobertura_card, ~70%)
    lateral.joblib    -> probabilidade de dia lateral, ao lado da taxa dos ultimos 252 pregoes


A direcao do fechamento NAO e prevista (Fase D: nenhum modelo bate o retorno zero).

Saida: data/previsoes/AAAA-MM-DD.json e AAAA-MM-DD_card.txt (nunca sobrescreve; uma
nova execucao no mesmo dia ganha o sufixo _HHMMSS). Em dia sem pregao na B3, grava
so o registro "sem_pregao".

Uso:
    python src/predictor_9h15.py
    python src/predictor_9h15.py --data 2026-10-16 --simulacao [--snapshots <pasta>]
        (--simulacao grava em data/simulacao/ em vez de data/)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import dataset_9h15 as ds

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).parent.parent
DIR_MODELOS = RAIZ / "models"
JANELA_TAXA_LATERAL = 252          # mesma do treino_walkforward.py
LIMIAR_LATERAL = ds.LIMIAR_LATERAL
EPS = 1e-5


def pastas(simulacao: bool) -> dict:
    base = RAIZ / "data" / ("simulacao" if simulacao else "")
    return {"previsoes": base / "previsoes", "historico": base / "historico_previsoes.csv"}


def aplicar(modelo: dict, x: pd.DataFrame) -> float:
    """Aplica um modelo salvo pelo treino_walkforward.py a uma linha de features."""
    tipo = modelo["tipo"]
    if tipo == "coluna":
        return float(x[modelo["features"][0]].iloc[0])
    if tipo == "zero":
        return 0.0
    if tipo == "media":
        return float(modelo["estimador"])
    xs = x[modelo["features"]]
    if modelo["transformacao_log"]:
        xs = np.log(xs.clip(lower=EPS))
    if modelo["saida"] == "probabilidade":
        return float(modelo["estimador"].predict_proba(xs)[0, 1])
    p = float(modelo["estimador"].predict(xs)[0])
    return float(np.exp(p)) if modelo["transformacao_log"] else p


def linha_do_dia(dados: dict, t: pd.Timestamp, dir_snapshots: Path) -> dict:
    """Features e referencias do pregao t, so com o que era conhecido as 9h15.

    Overnight em duas versoes: candle de 1h das 9h (igual ao treino) e snapshot das
    9h15 (quando existir).
    """
    trunc = ds.truncar(dados, t, "9h15")
    vol = ds.montar_vol(trunc)
    taxa_252 = vol["lateral"].rolling(JANELA_TAXA_LATERAL, min_periods=200).mean().shift(1)
    b = ds.bova11_diario(trunc)
    anteriores = b[(b.index < t) & b["Close"].notna()]

    on_9h = ds.montar_gap(trunc, "9h").loc[t, ds.FEATS_GAP].to_dict()
    snapshot = dir_snapshots / f"{t:%Y-%m-%d}_9h15.csv"
    on_9h15 = ds.features_snapshot_dia(snapshot) if snapshot.exists() else {}
    if pd.isna(on_9h15.get("var_on_EWZ", np.nan)):
        on_9h15 = {}

    return {
        "features": {**vol.loc[t, ds.FEATS_VOL].to_dict(), "taxa_lateral_252d": float(taxa_252.loc[t])},
        "overnight_9h": on_9h,
        "overnight_9h15": on_9h15,
        "snapshot_9h15": snapshot.name if on_9h15 else None,
        "fech_anterior": float(anteriores["Close"].iloc[-1]),
        "data_fech_anterior": str(anteriores.index[-1].date()),
        "fonte_fech_anterior": anteriores["fonte_barra"].iloc[-1],
    }


def prever(t: pd.Timestamp, dir_snapshots: Path, dados: dict = None) -> dict:
    dados = dados or ds.baixar_dados()
    if t not in dados["sessoes"]:
        return {"data": str(t.date()), "status": "sem_pregao"}

    lin = linha_do_dia(dados, t, dir_snapshots)
    modelos = {a: joblib.load(DIR_MODELOS / f"{a}.joblib") for a in ("gap", "amplitude", "lateral")}
    faixa = json.loads((DIR_MODELOS / "faixa.json").read_text(encoding="utf-8"))
    x = pd.DataFrame([lin["features"]])
    amp = aplicar(modelos["amplitude"], x)
    p_lat = aplicar(modelos["lateral"], x)
    gap_9h = aplicar(modelos["gap"], pd.DataFrame([lin["overnight_9h"]]))
    gap_9h15 = aplicar(modelos["gap"], pd.DataFrame([lin["overnight_9h15"]])) if lin["overnight_9h15"] else None

    # o card usa a previsao com o EWZ das 9h15; sem snapshot, a das 9h
    gap = gap_9h15 if gap_9h15 is not None else gap_9h
    abertura = lin["fech_anterior"] * (1 + gap)
    f = lin["features"]
    return {
        "data": str(t.date()), "status": "ok",
        "gerado_em": pd.Timestamp.now(tz=ds.TZ).strftime("%Y-%m-%d %H:%M:%S%z"),
        "fech_anterior": lin["fech_anterior"], "data_fech_anterior": lin["data_fech_anterior"],
        "fonte_fech_anterior": lin["fonte_fech_anterior"],
        "gap_prev_9h": gap_9h, "gap_prev_9h15": gap_9h15,
        "fonte_gap_card": f"snapshot 9h15 ({lin['snapshot_9h15']})" if gap_9h15 is not None
        else "candle de 1h das 9h (sem snapshot 9h15)",
        "gap_previsto": gap, "abertura_esperada": abertura,
        "amplitude_prevista": amp, "amplitude_prevista_rs": amp * abertura,
        "faixa_inferior": abertura * (1 - faixa["k_inf"] * amp),
        "faixa_superior": abertura * (1 + faixa["k_sup"] * amp),
        "faixa_k_inf": faixa["k_inf"], "faixa_k_sup": faixa["k_sup"],
        "faixa_cobertura_alvo": faixa["cobertura_alvo"], "faixa_cobertura_card": faixa["cobertura_card"],
        "prob_lateral": p_lat,
        "baselines": {"amplitude_har_5d": f["har_5d"], "gap_zero": 0.0,
                      "taxa_lateral_252d": f["taxa_lateral_252d"]},
        "modelos": {a: {"modelo": m["modelo"], "treino_ate": m["treino_ate"]} for a, m in modelos.items()},
        "features": f, "overnight_9h": lin["overnight_9h"], "overnight_9h15": lin["overnight_9h15"],
    }


def fmt_rs(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(v: float, sinal: bool = False) -> str:
    return (f"{v * 100:+.2f}%" if sinal else f"{v * 100:.2f}%").replace(".", ",")


def card_telegram(p: dict) -> str:
    data = pd.Timestamp(p["data"]).strftime("%d/%m/%Y")
    if p["status"] != "ok":
        return f"IOSI | BOVA11 | {data}\nSem pregao na B3 hoje."
    aviso_fech = "" if p["fonte_fech_anterior"] == "yahoo_1d" else " (aprox., candles de 1h)"
    ewz = "EWZ 9h15" if p["gap_prev_9h15"] is not None else "EWZ 9h"
    return "\n".join([
        f"📊 IOSI | BOVA11 | {data} (9h15)",
        f"Fech. anterior ({pd.Timestamp(p['data_fech_anterior']):%d/%m}): {fmt_rs(p['fech_anterior'])}{aviso_fech}",
        f"Abertura esperada: {fmt_rs(p['abertura_esperada'])} (gap {fmt_pct(p['gap_previsto'], True)}, base {ewz})",
        f"Amplitude esperada: {fmt_pct(p['amplitude_prevista'])} ≈ {fmt_rs(p['amplitude_prevista_rs'])}",
        # cobertura medida com a faixa centrada na abertura esperada (caso real do robo)
        f"Faixa provável (~{p['faixa_cobertura_card']:.0%} dos dias): {fmt_rs(p['faixa_inferior'])} a "
        f"{fmt_rs(p['faixa_superior'])}",
        f"Dia lateral (|fech−abert| < 1%): {p['prob_lateral']:.0%} (últimos 252 pregões: "
        f"{p['baselines']['taxa_lateral_252d']:.0%})",
        "⚠️ A direção do fechamento não é prevista.",
    ])


def caminho_livre(pasta: Path, nome: str, ext: str) -> Path:
    alvo = pasta / f"{nome}{ext}"
    return alvo if not alvo.exists() else pasta / f"{nome}_{pd.Timestamp.now(tz=ds.TZ):%H%M%S}{ext}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="pregao a prever (AAAA-MM-DD); padrao: hoje")
    parser.add_argument("--simulacao", action="store_true", help="grava em data/simulacao/")
    parser.add_argument("--snapshots", type=Path, default=ds.DIR_SNAPSHOTS)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    t = pd.Timestamp(args.data) if args.data else ds.HOJE
    destino = pastas(args.simulacao)["previsoes"]
    destino.mkdir(parents=True, exist_ok=True)

    previsao = prever(t, args.snapshots)
    previsao["simulacao"] = args.simulacao
    previsao["alerta_formatado_telegram"] = card_telegram(previsao)

    arq = caminho_livre(destino, f"{t:%Y-%m-%d}", ".json")
    with open(arq, "x", encoding="utf-8") as fh:
        json.dump(previsao, fh, indent=2, ensure_ascii=False, default=float)
    with open(arq.with_name(arq.stem + "_card.txt"), "x", encoding="utf-8") as fh:
        fh.write(previsao["alerta_formatado_telegram"] + "\n")
    logger.info("Previsao salva: %s", arq)
    print(previsao["alerta_formatado_telegram"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
