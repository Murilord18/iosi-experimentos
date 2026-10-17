"""
complemento_fase_a.py
=====================
Complemento da Fase A: HAR (amplitude media 1/5/22d) e nivel do VIX em t-1 contra
|close-open|/open, comparado com a correlacao contra a amplitude (high-low)/open.

Le correlacao_features_proprias.csv (gerado por correlacao_ativos.py) para manter os
mesmos dados e janelas da Fase A: 5 anos e ~2 anos (periodo dos candles de 1h).
Valores entre parenteses: |r| < 2/sqrt(n).
"""

import sys
from pathlib import Path

import pandas as pd

DIR = Path(__file__).parent.parent / "relatorios" / "correlacao_ativos"
FEATS = {"amp_media_1d": "HAR 1d", "amp_media_5d": "HAR 5d", "amp_media_22d": "HAR 22d",
         "vix_nivel_t1": "Nivel do VIX (t-1)"}
JANELAS = {"5 anos": "5 anos", "~2 anos": "2 anos"}
ALVOS = {"abs_intraday": "|close−open|/open", "amplitude": "amplitude"}


def fmt(r: float, lim: float) -> str:
    s = f"{r:.2f}".replace("-", "−").replace(".", ",")
    return s if abs(r) >= lim else f"({s})"


def main() -> None:
    c = pd.read_csv(DIR / "correlacao_features_proprias.csv")
    c = c[c["alvo_bova11"].isin(ALVOS) & c["ativo"].isin(FEATS)]
    c.to_csv(DIR / "complemento_abs_intraday.csv", index=False, float_format="%.6f")

    linhas = [r"| Feature (até t−1) | Janela | n | \|close−open\|/open (P / S) | amplitude (P / S) |"
              r" Razão P (\|c−o\| ÷ amp) |", "|---|---|---|---|---|---|"]
    for janela, rot_j in JANELAS.items():
        for f, rot_f in FEATS.items():
            sub = c[(c["ativo"] == f) & (c["janela"] == janela)].set_index("alvo_bova11")
            co, amp = sub.loc["abs_intraday"], sub.loc["amplitude"]
            linhas.append(
                f"| {rot_f} | {rot_j} | {int(co['n'])} "
                f"| {fmt(co['pearson'], co['limiar_2_raiz_n'])} / {fmt(co['spearman'], co['limiar_2_raiz_n'])} "
                f"| {fmt(amp['pearson'], amp['limiar_2_raiz_n'])} / {fmt(amp['spearman'], amp['limiar_2_raiz_n'])} "
                f"| {co['pearson'] / amp['pearson']:.2f} |".replace(".", ","))
    tabela = "\n".join(linhas)
    sys.stdout.reconfigure(encoding="utf-8")
    print(tabela)
    (DIR / "complemento_abs_intraday_tabela.md").write_text(tabela + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
