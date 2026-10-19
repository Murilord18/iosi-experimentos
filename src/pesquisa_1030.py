"""
pesquisa_1030.py
================
Projeto IOSI - pesquisa: os primeiros 30 minutos do pregao ajudam a prever o resto do dia?

a) Candles de 5m (~60 dias do Yahoo), features point-in-time das 10h30 (leitura_1030.py,
   so candles com fim <= 10h30):
       amp_ate_1030      (max - min ate 10h30) / abertura real
       abs_ret_ate_1030  |preco 10h30 / abertura - 1|      (e ret_ate_1030 com sinal)
       abs_gap_real      |abertura / fechamento anterior - 1|   (e gap_real com sinal)
       vol_rel_1030      volume 10h-10h30 / media do mesmo trecho nos 20 pregoes anteriores
       var_1030_<ativo>  ultimo preco ate 10h30 / fechamento diario anterior - 1 (EWZ, ES=F,
                         ^VIX; para EWZ e ES=F tambem o valor absoluto)
   x alvos do resto do dia (candles que comecam depois do preco das 10h30):
       amp_resto         (max - min depois de 10h30) / preco 10h30
       abs_ret_resto     |close / preco 10h30 - 1|
       lateral_resto     abs_ret_resto < 1%
       amplitude         (high - low) / open do dia inteiro
   Pearson e Spearman (insignificante: |r| < 2/sqrt(n)) e correlacao parcial controlando
   log(HAR 1/5/22d) e log(VIX).

b) PROXY com ~2 anos (candles de 1h): o candle das 10h-11h no lugar dos primeiros 30 min.
   Usa informacao ate 11h, entao o resto do dia passa a ser depois de 11h
   (amp_resto_proxy = (max - min dos candles >= 11h) / close das 11h). Walk-forward mensal,
   OLS em log:
       HAR+VIX            log(har_1d, har_5d, har_22d, vix_nivel_t1)   (modelo atual)
       HAR+VIX+amp_1h     + log((high - low) / open do candle das 10h)
   contra har_5d e har_22d (para o resto do dia, escalados pela razao mediana alvo/har do
   treino, ja que o resto do dia e menor que o dia inteiro). Curva de aprendizado: o mesmo
   com janela movel de N pregoes, para estimar quantos dias de snapshots das 10h30 seriam
   necessarios.

c) Conclusao no relatorios/pesquisa_1030/resumo.md. O modelo de producao NAO muda.

Uso:
    python src/pesquisa_1030.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

import dataset_9h15 as ds
import leitura_1030 as l1030

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).parent.parent
DIR_REL = RAIZ / "relatorios" / "pesquisa_1030"
HAR_VIX = ["har_1d", "har_5d", "har_22d", "vix_nivel_t1"]
DIA_ATIPICO = pd.Timestamp("2026-10-05")
JANELA_VOLUME = 20
MIN_TREINO = 120
JANELAS_CURVA = [60, 120, 250, None]          # None = expansiva
EPS = 1e-5

FEATS_5M = {
    "amp_ate_1030": "amplitude dos primeiros 30 min",
    "abs_ret_ate_1030": "|retorno dos primeiros 30 min|",
    "ret_ate_1030": "retorno dos primeiros 30 min",
    "abs_gap_real": "|gap real|",
    "gap_real": "gap real",
    "vol_rel_1030": "volume 10h-10h30 / média 20 pregões",
    "abs_var_1030_EWZ": "|variação EWZ até 10h30|",
    "var_1030_EWZ": "variação EWZ até 10h30",
    "abs_var_1030_ES_FUT": "|variação ES=F até 10h30|",
    "var_1030_ES_FUT": "variação ES=F até 10h30",
    "var_1030_VIX": "variação VIX até 10h30",
    "har_5d": "referência: HAR 5d",
    "har_22d": "referência: HAR 22d",
    "vix_nivel_t1": "referência: nível do VIX (t−1)",
}
ALVOS_5M = {"amp_resto": "amp_resto", "abs_ret_resto": "|ret_resto|", "lateral_resto": "lateral_resto",
            "amplitude": "amplitude do dia"}


# --------------------------------------------------------------------------- #
# a) 5 minutos
# --------------------------------------------------------------------------- #
def dataset_5m(dados: dict, vol: pd.DataFrame) -> pd.DataFrame:
    candles = {"BOVA11": l1030.de_yahoo(ds.baixar("BOVA11.SA", period="60d", interval="5m"))}
    for nome, tk in l1030.EXTERNOS.items():
        candles[nome] = l1030.de_yahoo(ds.baixar(tk, period="60d", interval="5m", prepost=True))
    b = ds.bova11_diario(dados)
    c5 = candles["BOVA11"]
    dia_c5 = c5["inicio"].dt.tz_localize(None).dt.normalize()

    linhas = []
    for t in sorted(dia_c5.unique()):
        t = pd.Timestamp(t)
        if t not in b.index or pd.isna(b.loc[t, "Close"]):
            continue
        anteriores = b[(b.index < t) & b["Close"].notna()]
        fech = {"BOVA11": float(anteriores["Close"].iloc[-1])}
        for nome in l1030.EXTERNOS:
            d = dados["diario"][nome]["Close"].dropna()
            fech[nome] = float(d[d.index < t].iloc[-1])
        f = l1030.features_1030(candles, fech, t)
        lb = l1030.leitura_bova(c5, t)
        if not lb:
            logger.warning("%s: sem candles do BOVA11 ate 10h30 no Yahoo; dia fora da amostra", t.date())
            continue
        depois = c5[(dia_c5 == t) & (c5["inicio"] >= lb["preco_1030_fim"])]
        ret = b.loc[t, "Close"] / lb["preco_1030"] - 1
        linhas.append({"pregao": t, **f, "n_candles_ate_1030": lb["n_candles_ate_1030"],
                       "amp_resto": (depois["high"].max() - depois["low"].min()) / lb["preco_1030"],
                       "ret_resto": ret, "abs_ret_resto": abs(ret),
                       "lateral_resto": float(abs(ret) < ds.LIMIAR_LATERAL)})
    df = pd.DataFrame(linhas).set_index("pregao")
    df["vol_rel_1030"] = df["volume_ate_1030"] / df["volume_ate_1030"].rolling(
        JANELA_VOLUME, min_periods=5).mean().shift(1)
    for c in ("ret_ate_1030", "gap_real", "var_1030_EWZ", "var_1030_ES_FUT"):
        df[f"abs_{c}"] = df[c].abs()
    return df.join(vol[HAR_VIX + ["amplitude", "abs_intraday", "lateral"]])


def fmt_r(r: float, n: int) -> str:
    if not np.isfinite(r):
        return "—"
    s = f"{r:.2f}".replace(".", ",")
    return s if abs(r) >= 2 / np.sqrt(n) else f"({s})"


def residuo(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    m = LinearRegression().fit(x, y)
    return y - m.predict(x)


def correlacoes(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson, Spearman, Pearson sem 05/10/2026 e parcial (controle log HAR+VIX)."""
    linhas = []
    for alvo in ALVOS_5M:
        for feat in FEATS_5M:
            sub = df[list(dict.fromkeys([feat, alvo] + HAR_VIX))].dropna()
            n = len(sub)
            if n < 10:
                continue
            sem = sub[sub.index != DIA_ATIPICO]
            linha = {"alvo": alvo, "feature": feat, "n": n, "limiar": 2 / np.sqrt(n),
                     "pearson": sub[feat].corr(sub[alvo]),
                     "spearman": spearmanr(sub[feat], sub[alvo]).statistic,
                     "pearson_sem_05_10": sem[feat].corr(sem[alvo]), "n_sem_05_10": len(sem)}
            if feat not in HAR_VIX:
                ctrl = np.log(sub[HAR_VIX].clip(lower=EPS))
                positiva = (sub[feat] > 0).all() and feat not in ("ret_ate_1030", "gap_real")
                xf = np.log(sub[feat].clip(lower=EPS)) if positiva else sub[feat]
                ya = np.log(sub[alvo].clip(lower=EPS)) if alvo in ("amp_resto", "amplitude") else sub[alvo]
                linha["parcial_har_vix"] = residuo(xf, ctrl).corr(residuo(ya, ctrl))
            linhas.append(linha)
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# b) proxy com o candle de 1h das 10h
# --------------------------------------------------------------------------- #
def dataset_proxy(dados: dict, vol: pd.DataFrame) -> pd.DataFrame:
    h = dados["h1"]["BOVA11"].copy()
    h["dia"] = h["inicio"].dt.tz_localize(None).dt.normalize()
    h["hora"] = h["inicio"].dt.hour
    primeira = h[h["hora"] == 10].drop_duplicates("dia").set_index("dia")
    resto = h[h["hora"] >= 11].groupby("dia").agg(high_resto=("High", "max"), low_resto=("Low", "min"),
                                                 n_resto=("High", "size"))
    df = pd.DataFrame({"amp_1h": (primeira["High"] - primeira["Low"]) / primeira["Open"],
                       "close_11h": primeira["Close"]}).join(resto, how="inner")
    df["amp_resto_proxy"] = (df["high_resto"] - df["low_resto"]) / df["close_11h"]
    df = df[df["n_resto"] >= 5].join(vol[HAR_VIX + ["amplitude"]], how="inner")
    df.index.name = "pregao"
    return df.dropna(subset=["amp_1h", "amp_resto_proxy", "amplitude"] + HAR_VIX)


def ols_log(tr: pd.DataFrame, te: pd.DataFrame, cols: list, alvo: str) -> np.ndarray:
    lx = lambda d: np.log(d[cols].clip(lower=EPS))           # noqa: E731
    m = LinearRegression().fit(lx(tr), np.log(tr[alvo].clip(lower=EPS)))
    return np.exp(m.predict(lx(te)))


def walk_forward(df: pd.DataFrame, alvo: str, janela: int = None, inicio_teste: pd.Timestamp = None) -> pd.DataFrame:
    """Mensal: treina com os pregoes ate o mes anterior (ou os ultimos `janela`) e preve o mes."""
    meses = df.index.to_period("M")
    out = []
    for m in meses.unique():
        tr, te = df[meses < m], df[meses == m]
        if len(tr) < MIN_TREINO or (inicio_teste is not None and te.index[0] < inicio_teste):
            continue
        if janela:
            tr = tr.tail(janela)
        p = pd.DataFrame({alvo: te[alvo]}, index=te.index)
        p["HAR+VIX"] = ols_log(tr, te, HAR_VIX, alvo)
        p["HAR+VIX+amp_1h"] = ols_log(tr, te, HAR_VIX + ["amp_1h"], alvo)
        for har in ("har_5d", "har_22d"):
            if alvo == "amplitude":
                p[har] = te[har]
            else:                                        # resto do dia: har escalado no treino
                p[f"{har}_escalado"] = te[har] * (tr[alvo] / tr[har]).median()
        out.append(p)
    return pd.concat(out)


def metricas(prev: pd.DataFrame, alvo: str) -> pd.DataFrame:
    linhas = []
    for periodo, sub in (("completo", prev), ("sem_05_10_2026", prev[prev.index != DIA_ATIPICO])):
        y = sub[alvo]
        modelos = [c for c in sub.columns if c != alvo]
        bases = [c for c in modelos if c.startswith("har_")]
        erro_a = (y - sub["HAR+VIX"]).abs()
        for mod in modelos:
            e = (y - sub[mod]).abs()
            d = erro_a - e
            linha = {"alvo": alvo, "periodo": periodo, "modelo": mod, "n": len(sub),
                     "de": str(sub.index.min().date()), "ate": str(sub.index.max().date()),
                     "mae_pp": e.mean() * 100, "spearman": spearmanr(sub[mod], y).statistic,
                     "ganho_mae_vs_HAR+VIX_%": (1 - e.mean() / erro_a.mean()) * 100,
                     "t_ganho_vs_HAR+VIX": d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if mod != "HAR+VIX" else np.nan}
            for base in bases:
                linha[f"ganho_mae_vs_{base}_%"] = (1 - e.mean() / (y - sub[base]).abs().mean()) * 100
            linhas.append(linha)
    return pd.DataFrame(linhas)


def curva_aprendizado(df: pd.DataFrame, alvo: str) -> pd.DataFrame:
    """Ganho do HAR+VIX+amp_1h sobre o HAR+VIX com janela movel de N pregoes, nos mesmos dias de teste."""
    inicio = df.index[max(JANELAS_CURVA[:-1])]
    linhas = []
    for janela in JANELAS_CURVA:
        p = walk_forward(df, alvo, janela, inicio_teste=inicio)
        ea, eb = (p[alvo] - p["HAR+VIX"]).abs(), (p[alvo] - p["HAR+VIX+amp_1h"]).abs()
        linhas.append({"alvo": alvo, "janela_treino": janela or "expansiva", "n_teste": len(p),
                       "de": str(p.index.min().date()), "mae_HAR+VIX_pp": ea.mean() * 100,
                       "mae_HAR+VIX+amp_1h_pp": eb.mean() * 100,
                       "ganho_%": (1 - eb.mean() / ea.mean()) * 100})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# Relatorio
# --------------------------------------------------------------------------- #
def num(v: float, casas: int = 2) -> str:
    return "—" if not np.isfinite(v) else f"{v:.{casas}f}".replace(".", ",")


def tabela_correlacoes(cor: pd.DataFrame) -> list:
    out = []
    for alvo, nome in ALVOS_5M.items():
        sub = cor[cor["alvo"] == alvo]
        if sub.empty:
            continue
        n = int(sub["n"].max())
        out += [f"\n**Alvo: {nome}** (n = {n}; insignificante se |r| < 2/√n ≈ {num(2 / np.sqrt(n))}, entre parênteses)\n",
                "| Feature | Pearson | Spearman | Pearson sem 05/10 | Parcial (controle HAR+VIX) |",
                "|---|---|---|---|---|"]
        for r in sub.itertuples():
            parcial = getattr(r, "parcial_har_vix", np.nan)
            out.append(f"| {FEATS_5M[r.feature].replace('|', chr(92) + '|')} | {fmt_r(r.pearson, r.n)} | {fmt_r(r.spearman, r.n)} | "
                       f"{fmt_r(r.pearson_sem_05_10, r.n_sem_05_10)} | {fmt_r(parcial, r.n)} |")
    return out


def tabela_metricas(met: pd.DataFrame, alvo: str, titulo: str) -> list:
    sub = met[(met["alvo"] == alvo) & (met["periodo"] == "completo")]
    bases = [c for c in sub.columns if c.startswith("ganho_mae_vs_har") and sub[c].notna().any()]
    r0 = sub.iloc[0]
    out = [f"\n**{titulo}** (n = {r0['n']}, {r0['de']} a {r0['ate']})\n",
           "| Modelo | MAE (pp) | Spearman | Ganho vs HAR+VIX | t do ganho | "
           + " | ".join(f"Ganho vs {b.removeprefix('ganho_mae_vs_').removesuffix('_%')}" for b in bases) + " |",
           "|---|---|---|---|---|" + "---|" * len(bases)]
    for r in sub.to_dict("records"):
        out.append(f"| {r['modelo']} | {num(r['mae_pp'], 3)} | {num(r['spearman'])} | "
                   f"{num(r['ganho_mae_vs_HAR+VIX_%'], 1)}% | {num(r['t_ganho_vs_HAR+VIX'], 1)} | "
                   + " | ".join(f"{num(r[b], 1)}%" for b in bases) + " |")
    sem = met[(met["alvo"] == alvo) & (met["periodo"] == "sem_05_10_2026") & (met["modelo"] == "HAR+VIX+amp_1h")]
    out.append(f"\nSem 05/10/2026: ganho do HAR+VIX+amp_1h sobre o HAR+VIX = "
               f"{num(sem['ganho_mae_vs_HAR+VIX_%'].iloc[0], 1)}% (t = {num(sem['t_ganho_vs_HAR+VIX'].iloc[0], 1)}).")
    return out


def dias_necessarios(r: float) -> float:
    """Pregoes para |r| passar de 2/sqrt(n): n = 4 / r^2."""
    return 4 / r ** 2 if r else np.inf


def dias_teste_t2(t: float, n: int) -> float:
    """Dias de teste para o ganho de MAE chegar a t = 2, se o efeito se mantiver (t cresce com sqrt(n))."""
    return n * (2 / t) ** 2 if t > 0 else np.inf


def conclusao(ganho: dict, curva: pd.DataFrame, rp_5m: float, rp_5m_dia: float, parcial_proxy: dict,
              n5: int) -> list:
    gr, gd = ganho["amp_resto_proxy"], ganho["amplitude"]
    sig = lambda t: "significativo" if t >= 2 else "não significativo"        # noqa: E731
    cr = curva[curva["alvo"] == "amp_resto_proxy"].set_index("janela_treino")["ganho_%"]
    cd = curva[curva["alvo"] == "amplitude"].set_index("janela_treino")["ganho_%"]
    har22 = gr.get("ganho_mae_vs_har_22d_escalado_%", np.nan)
    n_dia = dias_necessarios(rp_5m_dia)
    return [
        f"- **Resto do dia (amp_resto): a informação do início do pregão quase não ajuda.** No proxy de 2 anos, "
        f"acrescentar a amplitude da 1ª hora ao HAR+VIX reduz o MAE em {num(gr['ganho_mae_vs_HAR+VIX_%'], 1)}% "
        f"(t = {num(gr['t_ganho_vs_HAR+VIX'], 1)}, {sig(gr['t_ganho_vs_HAR+VIX'])}), e o modelo com a 1ª hora "
        f"fica {'atrás' if har22 < 0 else 'à frente'} do har_22d escalado ({num(har22, 1)}%). Com os 30 min reais "
        f"(n = {n5}), nenhuma feature passa do limiar de {num(2 / np.sqrt(n5))} na correlação parcial com a amp_resto "
        f"(amplitude dos 30 min: {num(rp_5m)}), e nem o HAR passa nessa amostra curta. Para |ret_resto| e "
        "lateral_resto, nada é significativo: a direção e o tamanho do movimento até o fechamento continuam "
        "imprevisíveis com o que se sabe às 10h30.",
        f"- **Dia inteiro (amplitude e faixa): ajuda bastante.** O ganho do proxy é de "
        f"{num(gd['ganho_mae_vs_HAR+VIX_%'], 1)}% (t = {num(gd['t_ganho_vs_HAR+VIX'], 1)}) e cresce com a janela de "
        f"treino ({num(cd.iloc[0], 1)}% com {cd.index[0]} pregões, {num(cd.iloc[-2], 1)}% com {cd.index[-2]}, "
        f"{num(cd.iloc[-1], 1)}% com a janela expansiva). Parte é mecânica (a 1ª hora faz parte do dia), mas é "
        "exatamente o que interessa à faixa das 10h30, que precisa conter a máxima e a mínima do dia. Com os 30 min "
        f"reais, a correlação parcial é {num(rp_5m_dia)} (n = {n5}, ainda abaixo do limiar), menor que a do proxy "
        f"de 1h ({num(parcial_proxy['amplitude'])}), como esperado para meia hora de pregão.",
        f"- **Quantos dias de snapshots das 10h30:** (1) para o efeito dos 30 min sobre a amplitude do dia ficar "
        f"significativo (|r| > 2/√n), ≈ {n_dia:.0f} pregões com r = {num(rp_5m_dia)}; (2) o walk-forward mensal "
        f"precisa de {MIN_TREINO} pregões de treino antes do primeiro mês de teste, e a curva de aprendizado mostra "
        f"o ganho ainda subindo de {cd.index[1]} para {cd.index[2]} pregões. **Recomendação: ~{max(MIN_TREINO, 250)} "
        "pregões (cerca de 1 ano) de snapshots das 10h30 para treinar e validar o modelo definitivo da amplitude do "
        f"dia/faixa; com ~{max(MIN_TREINO, int(round(n_dia, -1)))} já dá para uma primeira avaliação.** Para a "
        f"amp_resto, se o ganho do proxy ({num(gr['ganho_mae_vs_HAR+VIX_%'], 1)}%) for real, seriam necessários "
        f"≈ {dias_teste_t2(gr['t_ganho_vs_HAR+VIX'], int(gr['n'])):.0f} dias só de teste para confirmá-lo (t = 2), "
        "o que não compensa: o melhor uso dos primeiros 30 min é na faixa do dia inteiro.",
        f"- Curva de aprendizado da amp_resto: {', '.join(f'{num(v, 1)}% ({k})' for k, v in cr.items())}, sem "
        "tendência clara (ruído).",
        "- Os ~60 dias de candles de 5 min do Yahoo usados aqui (`dataset_5m_1030.csv`) adiantam o histórico: somados "
        "aos snapshots diários das 10h30, a contagem começa com eles. Como o Yahoo só guarda 60 dias, eles se perdem "
        "se não forem gravados: vale manter uma cópia deste CSV.",
    ]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DIR_REL.mkdir(parents=True, exist_ok=True)
    dados = ds.baixar_dados()
    vol = ds.montar_vol(dados)

    d5 = dataset_5m(dados, vol)
    cor = correlacoes(d5)
    dp = dataset_proxy(dados, vol)
    prev = {a: walk_forward(dp, a) for a in ("amp_resto_proxy", "amplitude")}
    met = pd.concat([metricas(p, a) for a, p in prev.items()])
    curva = pd.concat([curva_aprendizado(dp, a) for a in prev])
    # correlacao parcial do proxy (amostra toda) para comparar com a dos 30 min
    ctrl = np.log(dp[HAR_VIX].clip(lower=EPS))
    parcial_proxy = {a: residuo(np.log(dp["amp_1h"]), ctrl).corr(residuo(np.log(dp[a].clip(lower=EPS)), ctrl))
                     for a in prev}

    d5.to_csv(DIR_REL / "dataset_5m_1030.csv", index_label="pregao", float_format="%.10f")
    dp.to_csv(DIR_REL / "dataset_proxy_1h.csv", index_label="pregao", float_format="%.10f")
    cor.to_csv(DIR_REL / "correlacoes_5m.csv", index=False, float_format="%.6f")
    met.to_csv(DIR_REL / "walkforward_proxy_metricas.csv", index=False, float_format="%.6f")
    curva.to_csv(DIR_REL / "curva_aprendizado_proxy.csv", index=False, float_format="%.6f")
    pd.concat({a: p for a, p in prev.items()}, names=["alvo"]).to_csv(
        DIR_REL / "previsoes_proxy.csv", float_format="%.10f")

    # numeros da conclusao
    def c5(alvo, feat, col):
        r = cor[(cor["alvo"] == alvo) & (cor["feature"] == feat)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    ganho = {a: met[(met["alvo"] == a) & (met["periodo"] == "completo") & (met["modelo"] == "HAR+VIX+amp_1h")].iloc[0]
             for a in prev}
    n5 = len(d5.dropna(subset=["amp_ate_1030", "amp_resto"] + HAR_VIX))
    rp_5m = c5("amp_resto", "amp_ate_1030", "parcial_har_vix")
    rp_5m_dia = c5("amplitude", "amp_ate_1030", "parcial_har_vix")
    rp_proxy = parcial_proxy["amp_resto_proxy"]

    md = [
        "# Pesquisa: os primeiros 30 minutos do pregão ajudam a prever o resto do dia?",
        "",
        f"Gerado por `src/pesquisa_1030.py` em {pd.Timestamp.now(tz=ds.TZ):%d/%m/%Y %H:%M}. "
        "O modelo de produção **não** foi trocado: aqui só há resultados.",
        "",
        "## a) Candles de 5 min (point-in-time às 10h30)",
        "",
        f"- Amostra: {n5} pregões com candles de 5 min do Yahoo ({d5.index.min():%d/%m/%Y} a {d5.index.max():%d/%m/%Y}). "
        "Todas as features usam só candles com fim ≤ 10h30 (`leitura_1030.py`, o mesmo código do `predictor_1030.py`, "
        "coberto pelo teste point-in-time).",
        "- No Yahoo, o 1º candle de 5 min do BOVA11 costuma ser o das 10h05 (o leilão de abertura do ETF termina "
        "por volta de 10h08). Assim, os \"primeiros 30 min\" são, na prática, ~22 min de negociação "
        f"(mediana de {d5['n_candles_ate_1030'].median():.0f} candles).",
        "- Resto do dia: candles que começam depois do preço das 10h30 (até 16h55; o leilão de fechamento não "
        "está nos candles de 5 min, mas o close é o da barra diária).",
        "- Correlação parcial: resíduo da feature e do alvo (log para a amplitude) depois de regredir os dois em "
        "log(HAR 1/5/22d) e log(VIX). Mede o que a feature acrescenta ao modelo atual.",
        *tabela_correlacoes(cor),
        "",
        "## b) Proxy com 2 anos: candle de 1h das 10h–11h",
        "",
        "> ⚠️ **Proxy:** o candle de 1h das 10h usa informação **até 11h**, não até 10h30. Por isso o \"resto do dia\" "
        "aqui começa às 11h (`amp_resto_proxy` = (máx. − mín. dos candles de 1h a partir de 11h) / close das 11h). "
        "A amplitude do dia inteiro inclui a 1ª hora, então o ganho para ela é em parte mecânico.",
        "",
        f"- Amostra: {len(dp)} pregões ({dp.index.min():%d/%m/%Y} a {dp.index.max():%d/%m/%Y}); walk-forward mensal, "
        f"janela expansiva, mínimo de {MIN_TREINO} pregões de treino; OLS em log.",
        "- Baselines: `har_5d` e `har_22d` (para o resto do dia, multiplicados pela razão mediana alvo/har do treino, "
        "porque o resto do dia é menor que o dia inteiro).",
        "- t do ganho: média da diferença diária de erro absoluto (HAR+VIX − modelo) dividida pelo erro padrão "
        "(ingênuo, sem correção de autocorrelação).",
        *tabela_metricas(met, "amp_resto_proxy", "Alvo: amp_resto (proxy, depois de 11h)"),
        *tabela_metricas(met, "amplitude", "Alvo: amplitude do dia"),
        "",
        "**Curva de aprendizado** (janela móvel de N pregões; mesmos dias de teste em todas as linhas)",
        "",
        "| Alvo | Janela de treino | n teste | MAE HAR+VIX (pp) | MAE +amp_1h (pp) | Ganho |",
        "|---|---|---|---|---|---|",
        *[f"| {r['alvo']} | {r['janela_treino']} | {r['n_teste']} | {num(r['mae_HAR+VIX_pp'], 3)} | "
          f"{num(r['mae_HAR+VIX+amp_1h_pp'], 3)} | {num(r['ganho_%'], 1)}% |" for r in curva.to_dict("records")],
        "",
        f"Correlação parcial de log(amp_1h) com o alvo, controlando HAR+VIX (amostra toda): "
        f"amp_resto proxy {num(rp_proxy)}, amplitude do dia {num(parcial_proxy['amplitude'])}.",
        "",
        "## c) Conclusão",
        "",
        *conclusao(ganho, curva, rp_5m, rp_5m_dia, parcial_proxy, n5),
        "",
        "Arquivos: `dataset_5m_1030.csv`, `correlacoes_5m.csv`, `dataset_proxy_1h.csv`, `previsoes_proxy.csv`, "
        "`walkforward_proxy_metricas.csv`, `curva_aprendizado_proxy.csv`.",
    ]
    (DIR_REL / "resumo.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
