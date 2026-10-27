"""
calibrar_faixa.py
=================
Fase E - Projeto IOSI. Calibra a faixa provavel do dia a partir da amplitude prevista.

Usa as previsoes fora da amostra do walk-forward da Fase D (HAR+VIX OLS log, treino so
ate o mes anterior) e as barras reais do BOVA11:
    sobe  = (high - open) / open / amplitude_prevista
    desce = (open - low)  / open / amplitude_prevista
Os multiplicadores k_sup e k_inf sao quantis q de "sobe" e "desce", com o mesmo q
escolhido para que maxima E minima fiquem dentro da faixa em 80% dos dias:
    faixa = [abertura - k_inf * amp * abertura, abertura + k_sup * amp * abertura]

Cobertura fora da amostra: a cada mes M, os multiplicadores sao calibrados so com os
dias de meses anteriores e aplicados a M. Tambem e medida a cobertura com a faixa
centrada na abertura ESPERADA (modelo de gap, ~1 ano de walk-forward), que e o caso
real do robo as 9h15.

Saida: models/faixa.json (multiplicadores calibrados com todos os dias) e
relatorios/walkforward/faixa_cobertura.csv.

Previsao das 10h30 (--cobertura-1030): as 10h30 a abertura do dia ja e conhecida, entao
a faixa e centrada na abertura REAL e ainda estendida para conter a maxima e a minima
ja ocorridas ate 10h30. Esta opcao refaz o walk-forward com a abertura real (completo e
ultimos 2 anos), mede o efeito da extensao nos ~60 dias com candles de 5m do Yahoo e
SO ACRESCENTA ao models/faixa.json os campos cobertura_card_1030, cobertura_card_1030_origem
e cobertura_1030 (k_sup e k_inf nao mudam).

Uso:
    python src/calibrar_faixa.py                     # recalibra tudo (reescreve faixa.json)
    python src/calibrar_faixa.py --cobertura-1030    # so a cobertura da faixa das 10h30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import dataset_9h15 as ds
import leitura_1030 as l1030

RAIZ = Path(__file__).parent.parent
DIR_REL = RAIZ / "relatorios" / "walkforward"
COL_AMP = "HAR+VIX_ols_log"
COL_GAP = "ridge_EWZ"
COBERTURA_ALVO = 0.80
MIN_DIAS_CALIBRACAO = 250


def multiplicadores(sobe: pd.Series, desce: pd.Series, alvo: float = COBERTURA_ALVO) -> tuple:
    """Menor quantil comum q tal que sobe <= Q_sobe(q) e desce <= Q_desce(q) em `alvo` dos dias."""
    lo, hi = alvo, 1.0
    for _ in range(40):
        q = (lo + hi) / 2
        ks, ki = sobe.quantile(q), desce.quantile(q)
        if ((sobe <= ks) & (desce <= ki)).mean() >= alvo:
            hi = q
        else:
            lo = q
    return float(sobe.quantile(hi)), float(desce.quantile(hi)), hi


def cobertura(df: pd.DataFrame, centro: str) -> pd.DataFrame:
    """Aplica, mes a mes, multiplicadores calibrados so com os meses anteriores."""
    out = []
    for m in df["mes"].unique():
        tr, te = df[df["mes"] < m], df[df["mes"] == m]
        if len(tr) < MIN_DIAS_CALIBRACAO:
            continue
        ks, ki, _ = multiplicadores(tr["sobe"], tr["desce"])
        c = te[centro]
        sup, inf = c * (1 + ks * te["amp_prev"]), c * (1 - ki * te["amp_prev"])
        out.append(te.assign(k_sup=ks, k_inf=ki, faixa_inf=inf, faixa_sup=sup,
                             max_dentro=te["High"] <= sup, min_dentro=te["Low"] >= inf))
    r = pd.concat(out)
    r["ambos_dentro"] = r["max_dentro"] & r["min_dentro"]
    return r


def resumo(r: pd.DataFrame, nome: str) -> list:
    linhas = []
    for periodo, sub in (("completo", r), ("ultimos_2_anos", r[r.index > ds.hoje() - pd.DateOffset(years=2)])):
        linhas.append({"centro": nome, "periodo": periodo, "n": len(sub),
                       "de": str(sub.index.min().date()), "ate": str(sub.index.max().date()),
                       "cobertura_max_e_min": sub["ambos_dentro"].mean(),
                       "cobertura_max": sub["max_dentro"].mean(), "cobertura_min": sub["min_dentro"].mean(),
                       "largura_media_pct": ((sub["faixa_sup"] - sub["faixa_inf"]) / sub["Open"]).mean() * 100})
    return linhas


def base_walkforward(b: pd.DataFrame) -> pd.DataFrame:
    amp = pd.read_csv(DIR_REL / "previsoes_amplitude.csv", parse_dates=["pregao"], index_col="pregao")
    df = b[["Open", "High", "Low", "Close"]].assign(fech_ant=b["Close"].shift(1)).join(
        amp[COL_AMP].rename("amp_prev"), how="inner").dropna()
    df["sobe"] = (df["High"] - df["Open"]) / df["Open"] / df["amp_prev"]
    df["desce"] = (df["Open"] - df["Low"]) / df["Open"] / df["amp_prev"]
    df["mes"] = df.index.to_period("M")
    return df


def cobertura_1030() -> None:
    """Cobertura da faixa das 10h30: centrada na abertura real e estendida ate a max/min das 10h30."""
    b = ds.bova11_diario(ds.baixar_dados())
    r = cobertura(base_walkforward(b), "Open")
    tab = resumo(r, "abertura real")

    # extensao (max/min ate 10h30) so pode ser medida onde ha candles de 5m (~60 dias)
    c5 = l1030.de_yahoo(ds.baixar("BOVA11.SA", period="60d", interval="5m"))
    ext = []
    for t in r.index[r.index >= c5["inicio"].min().tz_localize(None).normalize()]:
        lb = l1030.leitura_bova(c5[c5["inicio"].dt.tz_localize(None).dt.normalize() == t], t)
        if not lb:
            continue
        x = r.loc[t]
        sup, inf = max(x["faixa_sup"], lb["high_ate_1030"]), min(x["faixa_inf"], lb["low_ate_1030"])
        ext.append({"pregao": t, "ambos_sem_extensao": bool(x["ambos_dentro"]),
                    "ambos_com_extensao": bool(x["High"] <= sup and x["Low"] >= inf),
                    "faixa_ajustada": bool(sup > x["faixa_sup"] or inf < x["faixa_inf"])})
    ext = pd.DataFrame(ext).set_index("pregao")
    tab.append({"centro": "abertura real + max/min ate 10h30", "periodo": "dias com candles de 5m",
                "n": len(ext), "de": str(ext.index.min().date()), "ate": str(ext.index.max().date()),
                "cobertura_max_e_min": ext["ambos_com_extensao"].mean(),
                "cobertura_sem_extensao_mesmos_dias": ext["ambos_sem_extensao"].mean(),
                "frac_dias_faixa_ajustada": ext["faixa_ajustada"].mean()})

    # rotulo: o menor entre o periodo completo e os ultimos 2 anos (sem a extensao, que so aumenta)
    card = min(tab[:2], key=lambda x: x["cobertura_max_e_min"])
    arq = RAIZ / "models" / "faixa.json"
    faixa = json.loads(arq.read_text(encoding="utf-8"))
    faixa["cobertura_card_1030"] = round(float(card["cobertura_max_e_min"]), 4)
    faixa["cobertura_card_1030_origem"] = (
        f"walk-forward, faixa centrada na abertura REAL, maxima e minima do dia dentro, periodo "
        f"{card['periodo']} ({card['de']} a {card['ate']}, n={card['n']}); a extensao ate a max/min ja "
        f"ocorridas as 10h30 so aumenta a cobertura (medida em {len(ext)} dias com candles de 5m)")
    faixa["cobertura_1030"] = [{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                                for k, v in linha.items()} for linha in tab]
    arq.write_text(json.dumps(faixa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(tab).round(4).to_string(index=False))
    print(f"\ncobertura_card_1030 = {faixa['cobertura_card_1030']} ({card['periodo']}) gravada em {arq}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cobertura-1030", action="store_true")
    if parser.parse_args().cobertura_1030:
        return cobertura_1030()
    b = ds.bova11_diario(ds.baixar_dados())
    amp = pd.read_csv(DIR_REL / "previsoes_amplitude.csv", parse_dates=["pregao"], index_col="pregao")
    gap = pd.read_csv(DIR_REL / "previsoes_gap.csv", parse_dates=["pregao"], index_col="pregao")

    df = b[["Open", "High", "Low", "Close"]].assign(fech_ant=b["Close"].shift(1)).join(
        amp[COL_AMP].rename("amp_prev"), how="inner").dropna()
    df["sobe"] = (df["High"] - df["Open"]) / df["Open"] / df["amp_prev"]
    df["desce"] = (df["Open"] - df["Low"]) / df["Open"] / df["amp_prev"]
    df["mes"] = df.index.to_period("M")
    df["abertura_esperada"] = df["fech_ant"] * (1 + gap[COL_GAP].reindex(df.index))

    r_open = cobertura(df, "Open")
    # com a abertura esperada: multiplicadores calibrados no historico (centro = open real),
    # aplicados ao ano de walk-forward do gap
    esp = r_open.dropna(subset=["abertura_esperada"]).copy()
    c = esp["abertura_esperada"]
    esp["faixa_sup"], esp["faixa_inf"] = c * (1 + esp["k_sup"] * esp["amp_prev"]), c * (1 - esp["k_inf"] * esp["amp_prev"])
    esp["max_dentro"], esp["min_dentro"] = esp["High"] <= esp["faixa_sup"], esp["Low"] >= esp["faixa_inf"]
    esp["ambos_dentro"] = esp["max_dentro"] & esp["min_dentro"]
    r_open_mesmo_ano = r_open.loc[esp.index]

    tab = pd.DataFrame(resumo(r_open, "abertura real") + resumo(esp, "abertura esperada (gap)")
                       + resumo(r_open_mesmo_ano, "abertura real, mesmo ano do gap"))
    tab = tab[~((tab["centro"] != "abertura real") & (tab["periodo"] == "ultimos_2_anos"))]
    pd.concat([r_open.assign(centro="abertura real"), esp.assign(centro="abertura esperada")]) \
        .drop(columns="mes").to_csv(DIR_REL / "faixa_cobertura.csv", index_label="pregao", float_format="%.10f")

    ks, ki, q = multiplicadores(df["sobe"], df["desce"])
    # o card mostra a cobertura medida com a abertura esperada (caso real do robo), nao o alvo
    r_esp = tab[tab["centro"] == "abertura esperada (gap)"].iloc[0]
    final = {
        "k_sup": ks, "k_inf": ki, "quantil_comum": q, "cobertura_alvo": COBERTURA_ALVO,
        "cobertura_card": round(float(r_esp["cobertura_max_e_min"]), 4),
        "cobertura_card_origem": f"cobertura fora da amostra (maxima e minima dentro) com a faixa centrada na "
                                 f"abertura ESPERADA, {r_esp['de']} a {r_esp['ate']}, n={r_esp['n']}; "
                                 f"multiplicadores calibrados para {COBERTURA_ALVO:.0%} com a abertura real",
        "formula": "faixa = [abertura_esperada * (1 - k_inf * amp_prev), abertura_esperada * (1 + k_sup * amp_prev)]",
        "modelo_amplitude": COL_AMP, "n_dias": len(df),
        "calibrado_de": str(df.index.min().date()), "calibrado_ate": str(df.index.max().date()),
        "cobertura_fora_da_amostra": tab.round(4).to_dict(orient="records"),
    }
    (RAIZ / "models" / "faixa.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Multiplicadores finais (todos os {len(df)} dias, {final['calibrado_de']} a {final['calibrado_ate']}): "
          f"k_sup = {ks:.4f}, k_inf = {ki:.4f} (quantil comum {q:.4f})")
    print("\nDistribuicao de sobe/desce (x amplitude prevista):")
    print(df[["sobe", "desce"]].describe(percentiles=[0.5, 0.8, 0.9]).round(3).T.to_string())
    print("\nCobertura fora da amostra:")
    print(tab.round(4).to_string(index=False))
    print("\nMultiplicadores mes a mes (amostra):")
    print(r_open.groupby(r_open.index.year)[["k_sup", "k_inf"]].last().round(3).T.to_string())


if __name__ == "__main__":
    main()
