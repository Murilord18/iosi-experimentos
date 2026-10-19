"""
predictor_1030.py
=================
Projeto IOSI. Previsao do dia as 10h30 (BRT), com a B3 ja aberta: a abertura do dia ja
e conhecida, e o foco e o RESTO DO DIA (das 10h30 ao fechamento).

Point-in-time: so usa barras diarias com data < t e candles de 5m com fim <= 10h30 de t.
    leitura das 10h30   data/snapshots/AAAA-MM-DD_10h30.csv (coletor modo manha_1030); sem
                        snapshot, os candles de 5m do Yahoo, cortados em fim <= 10h30
                        (leitura_1030.py): abertura real (open do 1o candle de 5m), preco as
                        10h30 (close do ultimo candle encerrado) e max/min ja ocorridas
    amplitude.joblib    amplitude esperada do dia (modelo atual, features ate t-1)
    faixa.json          faixa provavel centrada na abertura REAL:
                        [abertura x (1 - k_inf x amp), abertura x (1 + k_sup x amp)], estendida
                        para conter a maxima e a minima ja ocorridas ate 10h30; o card mostra a
                        cobertura do walk-forward com a abertura real (cobertura_card_1030,
                        calibrar_faixa.py --cobertura-1030)
    lateral.joblib      probabilidade de dia lateral (|close-open|/open < 1%), ao lado da taxa
                        dos ultimos 252 pregoes

A direcao do fechamento NAO e prevista. Os modelos ainda sao os das 9h15 (a pesquisa sobre
o inicio do pregao esta em pesquisa_1030.py); as features das 10h30 ficam gravadas no JSON
(features_1030) para treinar o modelo definitivo.

Saida: data/previsoes/AAAA-MM-DD_1030.json e AAAA-MM-DD_1030_card.txt (nunca sobrescreve;
uma nova execucao no mesmo dia ganha o sufixo _HHMMSS). A primeira previsao do dia (a que o
validador usa) tambem vai para a tabela previsoes do Supabase.

Uso:
    python src/predictor_1030.py
    python src/predictor_1030.py --data 2026-10-16 --simulacao [--snapshots <pasta>]
        (--simulacao grava em data/simulacao/ em vez de data/)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

import dataset_9h15 as ds
import leitura_1030 as l1030
import storage_supabase as sb
from predictor_9h15 import DIR_MODELOS, JANELA_TAXA_LATERAL, aplicar, caminho_livre, fmt_pct, fmt_rs, pastas

logger = logging.getLogger(__name__)

SUFIXO = "_1030"


def snapshot_do_dia(dir_snapshots: Path, t: pd.Timestamp) -> Path | None:
    """O snapshot das 10h30 do dia (o primeiro; se nao houver, o ultimo extra _HHMMSS)."""
    base = dir_snapshots / f"{t:%Y-%m-%d}_10h30.csv"
    if base.exists():
        return base
    extras = sorted(dir_snapshots.glob(f"{t:%Y-%m-%d}_10h30_*.csv"))
    return extras[-1] if extras else None


def candles_1030(t: pd.Timestamp, dir_snapshots: Path) -> tuple[dict, str]:
    """Candles de 5m de BOVA11 e dos externos para a leitura das 10h30, e a origem."""
    arq = snapshot_do_dia(dir_snapshots, t)
    if arq is not None:
        candles, _ = l1030.ler_snapshot(arq)
        if "BOVA11" in candles:
            return candles, f"snapshot {arq.name}"
        logger.warning("%s sem candles do BOVA11; usando o Yahoo", arq.name)
    logger.warning("Sem snapshot das 10h30 de %s: candles de 5m do Yahoo (fim <= 10h30)", t.date())
    candles = {"BOVA11": l1030.de_yahoo(ds.baixar("BOVA11.SA", period="60d", interval="5m"))}
    for nome, tk in l1030.EXTERNOS.items():
        try:
            candles[nome] = l1030.de_yahoo(ds.baixar(tk, period="60d", interval="5m", prepost=True))
        except RuntimeError as exc:
            logger.warning("%s", exc)
    return candles, "yahoo 5m (sem snapshot 10h30)"


def prever(t: pd.Timestamp, dir_snapshots: Path, dados: dict = None) -> dict:
    dados = dados or ds.baixar_dados()
    base = {"data": str(t.date()), "previsao_hora": "10h30"}
    if t not in dados["sessoes"]:
        return {**base, "status": "sem_pregao"}

    trunc = ds.truncar(dados, t, "10h30")           # barras diarias com data < t
    vol = ds.montar_vol(trunc)
    taxa_252 = vol["lateral"].rolling(JANELA_TAXA_LATERAL, min_periods=200).mean().shift(1)
    b = ds.bova11_diario(trunc)
    anteriores = b[(b.index < t) & b["Close"].notna()]
    fech_ant = {"BOVA11": float(anteriores["Close"].iloc[-1])}
    for nome in l1030.EXTERNOS:
        d = trunc["diario"][nome]["Close"].dropna()
        if len(d):
            fech_ant[nome] = float(d.iloc[-1])

    candles, fonte = candles_1030(t, dir_snapshots)
    ref = l1030.referencia(t)
    lb = l1030.leitura_bova(candles["BOVA11"], t, ref)
    if not lb:
        return {**base, "status": "sem_leitura_1030", "fonte_leitura_1030": fonte}
    f1030 = l1030.features_1030(candles, fech_ant, t)

    modelos = {a: joblib.load(DIR_MODELOS / f"{a}.joblib") for a in ("amplitude", "lateral")}
    faixa = json.loads((DIR_MODELOS / "faixa.json").read_text(encoding="utf-8"))
    f = {**vol.loc[t, ds.FEATS_VOL].to_dict(), "taxa_lateral_252d": float(taxa_252.loc[t])}
    x = pd.DataFrame([f])
    amp = aplicar(modelos["amplitude"], x)
    p_lat = aplicar(modelos["lateral"], x)

    abertura = lb["abertura_real"]
    modelo_inf, modelo_sup = abertura * (1 - faixa["k_inf"] * amp), abertura * (1 + faixa["k_sup"] * amp)
    # a faixa vai ate o fechamento: precisa conter o que ja aconteceu ate 10h30
    inf, sup = min(modelo_inf, lb["low_ate_1030"]), max(modelo_sup, lb["high_ate_1030"])
    return {
        **base, "status": "ok",
        "gerado_em": pd.Timestamp.now(tz=ds.TZ).strftime("%Y-%m-%d %H:%M:%S%z"),
        "fonte_leitura_1030": fonte,
        "snapshot_1030": fonte.removeprefix("snapshot ") if fonte.startswith("snapshot ") else None,
        "fech_anterior": fech_ant["BOVA11"], "data_fech_anterior": str(anteriores.index[-1].date()),
        "fonte_fech_anterior": anteriores["fonte_barra"].iloc[-1],
        "abertura_real": abertura, "gap_real": abertura / fech_ant["BOVA11"] - 1,
        "preco_1030": lb["preco_1030"], "preco_1030_fim": lb["preco_1030_fim"].strftime("%Y-%m-%d %H:%M:%S%z"),
        "idade_preco_1030_min": lb["idade_preco_1030_min"],
        "high_ate_1030": lb["high_ate_1030"], "low_ate_1030": lb["low_ate_1030"],
        "ret_ate_1030": lb["ret_ate_1030"], "amp_ate_1030": lb["amp_ate_1030"],
        "n_candles_ate_1030": lb["n_candles_ate_1030"],
        "amplitude_prevista": amp, "amplitude_prevista_rs": amp * abertura,
        "faixa_modelo_inf": modelo_inf, "faixa_modelo_sup": modelo_sup,
        "faixa_inferior": inf, "faixa_superior": sup,
        "faixa_ajustada": bool(inf < modelo_inf or sup > modelo_sup),
        "faixa_k_inf": faixa["k_inf"], "faixa_k_sup": faixa["k_sup"],
        "faixa_cobertura_alvo": faixa["cobertura_alvo"], "faixa_cobertura_card": faixa["cobertura_card_1030"],
        "prob_lateral": p_lat,
        "baselines": {"amplitude_har_5d": f["har_5d"], "taxa_lateral_252d": f["taxa_lateral_252d"]},
        "modelos": {a: {"modelo": m["modelo"], "treino_ate": m["treino_ate"]} for a, m in modelos.items()},
        "features": f, "features_1030": f1030,
    }


def card_telegram(p: dict) -> str:
    data = pd.Timestamp(p["data"]).strftime("%d/%m/%Y")
    if p["status"] == "sem_pregao":
        return f"IOSI | BOVA11 | {data}\nSem pregao na B3 hoje."
    if p["status"] != "ok":
        return f"IOSI | BOVA11 | {data} (10h30)\nSem candles do BOVA11 ate 10h30: previsao nao gerada."
    aviso_fech = "" if p["fonte_fech_anterior"] == "yahoo_1d" else " (aprox., candles de 1h)"
    fim = pd.Timestamp(p["preco_1030_fim"])
    hora = "10h30" if p["idade_preco_1030_min"] == 0 else f"{fim:%Hh%M} (último candle até 10h30)"
    return "\n".join([
        f"📊 IOSI | BOVA11 | {data} (leitura das 10h30)",
        f"Fech. anterior ({pd.Timestamp(p['data_fech_anterior']):%d/%m}): {fmt_rs(p['fech_anterior'])}{aviso_fech}",
        f"Abertura real: {fmt_rs(p['abertura_real'])} (gap {fmt_pct(p['gap_real'], True)})",
        f"Preço às {hora}: {fmt_rs(p['preco_1030'])} ({fmt_pct(p['ret_ate_1030'], True)} desde a abertura)",
        f"Máx./mín. até 10h30: {fmt_rs(p['high_ate_1030'])} / {fmt_rs(p['low_ate_1030'])}",
        f"Amplitude esperada do dia: {fmt_pct(p['amplitude_prevista'])} ≈ {fmt_rs(p['amplitude_prevista_rs'])}",
        # cobertura do walk-forward com a faixa centrada na abertura real
        f"Faixa provável até o fechamento (~{p['faixa_cobertura_card']:.0%} dos dias): "
        f"{fmt_rs(p['faixa_inferior'])} a {fmt_rs(p['faixa_superior'])}",
        f"Dia lateral (|fech−abert| < 1%): {p['prob_lateral']:.0%} (últimos 252 pregões: "
        f"{p['baselines']['taxa_lateral_252d']:.0%})",
        "⚠️ A direção do fechamento não é prevista.",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="pregao a prever (AAAA-MM-DD); padrao: hoje")
    parser.add_argument("--simulacao", action="store_true", help="grava em data/simulacao/")
    parser.add_argument("--snapshots", type=Path, default=ds.DIR_SNAPSHOTS)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with sb.Execucao("manha_1030", "previsao", ativo=not args.simulacao) as ex:
        codigo = gerar(args)
        if codigo:
            ex.falha("status sem_leitura_1030 (sem candles do BOVA11 ate 10h30)")
    return codigo


def gerar(args) -> int:
    t = pd.Timestamp(args.data) if args.data else ds.HOJE
    destino = pastas(args.simulacao)["previsoes"]
    destino.mkdir(parents=True, exist_ok=True)

    previsao = prever(t, args.snapshots)
    previsao["simulacao"] = args.simulacao
    previsao["alerta_formatado_telegram"] = card_telegram(previsao)

    arq = caminho_livre(destino, f"{t:%Y-%m-%d}{SUFIXO}", ".json")
    with open(arq, "x", encoding="utf-8") as fh:
        json.dump(previsao, fh, indent=2, ensure_ascii=False, default=float)
    with open(arq.with_name(arq.stem + "_card.txt"), "x", encoding="utf-8") as fh:
        fh.write(previsao["alerta_formatado_telegram"] + "\n")
    logger.info("Previsao salva: %s", arq)
    if not args.simulacao:
        if arq.name == f"{t:%Y-%m-%d}{SUFIXO}.json":
            sb.upsert_previsao(previsao, arq.name)
        else:
            logger.info("Previsao extra do dia (%s): nao enviada ao Supabase", arq.name)
    print(previsao["alerta_formatado_telegram"])
    return 0 if previsao["status"] in ("ok", "sem_pregao") else 1


if __name__ == "__main__":
    sys.exit(main())
