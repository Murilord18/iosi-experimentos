"""
correlacao_ativos.py
====================
Modulo 1 - Projeto IOSI (Indice de Oportunidade de Straddle Intraday)
Correlacao entre o BOVA11 e ativos globais (indices, moedas e futuros).

Etapa 1 - Diario (5 anos):
    - Correlacao de retornos com o BOVA11 (simultanea e defasada em 1 dia)
    - Correlacao com o |retorno| e com a amplitude do BOVA11
    - Matriz de correlacao de retornos entre os ativos
    - Features proprias: nivel do VIX em t-1 e amplitude media do BOVA11 (1, 5, 22 dias)

Etapa 2 - Overnight (candles de 1h com prepost=True, ~730 dias):
    - Preco de cada ativo as 9h BRT de cada pregao do BOVA11
    - Variacao overnight contra o fechamento anterior
    - Correlacao com gap, retorno de fechamento, retorno intradiario e amplitude do BOVA11
    - EWZ pre-mercado em candles de 5m (~60 dias): horario do ultimo negocio e 9h x 9h15
    - Correlacao parcial de USDBRL e DXY controlando pelo EWZ pre-mercado

Regras de dados:
    - Fuso sempre explicito (America/Sao_Paulo); so ffill, nunca bfill.
    - Correlacao sempre de retornos/variacoes, nunca de precos.
    - Correlacoes com |r| < 2/sqrt(n) sao marcadas como insignificantes.
    - Cambio (tickers =X): no Yahoo, o Close diario com data D e o preco das ~21h BRT
      de D-1. Na Etapa 1 a serie e recuada 1 barra; na Etapa 2 o fechamento anterior
      vem do candle de 1h no horario de fechamento da B3 em D-1.
"""

import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ATIVOS_DIARIOS = {
    "BOVA11":      "BOVA11.SA",
    "IBOV":        "^BVSP",
    "EWZ":         "EWZ",
    "SP500":       "^GSPC",
    "NASDAQ":      "^IXIC",
    "RUSSELL2000": "^RUT",
    "VIX":         "^VIX",
    "USDBRL":      "BRL=X",
    "DXY":         "DX-Y.NYB",
    "EURBRL":      "EURBRL=X",
    "GBPBRL":      "GBPBRL=X",
    "USDJPY":      "JPY=X",
    "EURUSD":      "EURUSD=X",
}
FUTUROS = {
    "ES_FUT":    "ES=F",
    "NQ_FUT":    "NQ=F",
    "BRENT_FUT": "BZ=F",
}
ALVO = "BOVA11"
ATIVOS_OVERNIGHT = {**{k: v for k, v in ATIVOS_DIARIOS.items() if k != ALVO}, **FUTUROS}

TZ = ZoneInfo("America/Sao_Paulo")
HOJE = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
HORA_9H = pd.Timedelta(hours=9)
HORA_9H15 = pd.Timedelta(hours=9, minutes=15)
FECHAMENTO_B3_PADRAO = pd.Timedelta(hours=17)
DIAS_INTRADAY = 729                         # limite do Yahoo para 1h: 730 dias
DIAS_5M = 59                                # limite do Yahoo para 5m: 60 dias
IDADE_MAX_FRESCO_H = 3.0                    # preco das 9h considerado "do dia"
FRACAO_MIN_FRESCO = 0.5                     # abaixo disso o ativo nao tem pre-mercado util
JANELAS_HAR = (1, 5, 22)
N_MIN = 30

OUT_DIR = Path(__file__).parent.parent / "relatorios" / "correlacao_ativos"

ROTULOS_ALVO = {
    "gap": "gap", "abs_gap": "|gap|",
    "ret_cc": "ret fech.", "abs_ret_cc": "|ret fech.|",
    "ret_intraday": "ret intrad.", "abs_intraday": "|close-open|/open",
    "amplitude": "amplitude",
}


# --------------------------------------------------------------------------- #
# Utilitarios
# --------------------------------------------------------------------------- #
def eh_cambio(ticker: str) -> bool:
    return ticker.endswith("=X")


def baixar(ticker: str, **kwargs) -> pd.DataFrame:
    for tentativa in range(1, 4):
        try:
            df = yf.download(ticker, progress=False, multi_level_index=False, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao baixar %s (tentativa %d): %s", ticker, tentativa, exc)
            df = pd.DataFrame()
        if not df.empty:
            return df.dropna(subset=["Close"])
    logger.error("Sem dados para %s", ticker)
    return pd.DataFrame()


def para_datas(idx: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def corrigir_datas_cambio(df: pd.DataFrame) -> pd.DataFrame:
    """Recua 1 barra a serie diaria de cambio do Yahoo.

    O Close da barra com data D e o preco das ~21h BRT de D-1. Recuar 1 dia de
    calendario perderia a sexta (nao ha barra de sabado: o fechamento de sexta so
    aparece na barra de segunda), por isso o recuo e de 1 barra. A barra do dia
    corrente traz o preco do momento e e descartada.
    """
    df = df[df.index < HOJE]
    return df.shift(-1).dropna(subset=["Close"])


def diario(ticker: str, **kwargs) -> pd.DataFrame:
    df = baixar(ticker, interval="1d", **kwargs)
    if df.empty:
        return df
    df.index = para_datas(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()[["Open", "High", "Low", "Close"]]
    return corrigir_datas_cambio(df) if eh_cambio(ticker) else df


def limiar(n: int) -> float:
    return 2 / np.sqrt(n) if n > 0 else np.nan


def correlacionar(x: pd.Series, y: pd.Series) -> dict:
    par = pd.concat([x, y], axis=1).dropna()
    n = len(par)
    if n < N_MIN or par.iloc[:, 0].std() == 0 or par.iloc[:, 1].std() == 0:
        return {"pearson": np.nan, "p_valor": np.nan, "spearman": np.nan, "n": n,
                "limiar_2_raiz_n": limiar(n), "significativo": False}
    r, p = stats.pearsonr(par.iloc[:, 0], par.iloc[:, 1])
    rho = stats.spearmanr(par.iloc[:, 0], par.iloc[:, 1]).statistic
    return {"pearson": r, "p_valor": p, "spearman": rho, "n": n,
            "limiar_2_raiz_n": limiar(n), "significativo": abs(r) >= limiar(n)}


def correlacao_parcial(x: pd.Series, y: pd.Series, controles: pd.DataFrame) -> dict:
    """Correlacao entre os residuos de x e y apos regressao (OLS) nos controles."""
    df = pd.concat([x.rename("_x"), y.rename("_y"), controles], axis=1).dropna()
    n, k = len(df), controles.shape[1]
    if n < N_MIN:
        return {"pearson": np.nan, "p_valor": np.nan, "n": n, "significativo": False}
    z = np.column_stack([np.ones(n), df[controles.columns].to_numpy()])
    res = {}
    for col in ("_x", "_y"):
        beta, *_ = np.linalg.lstsq(z, df[col].to_numpy(), rcond=None)
        res[col] = df[col].to_numpy() - z @ beta
    r = float(np.corrcoef(res["_x"], res["_y"])[0, 1])
    gl = n - 2 - k
    t = r * np.sqrt(gl / max(1e-12, 1 - r ** 2))
    return {"pearson": r, "p_valor": float(2 * stats.t.sf(abs(t), gl)), "n": n,
            "significativo": abs(r) >= limiar(n)}


def fmt_r(r: float, n: float) -> str:
    if pd.isna(r) or pd.isna(n):
        return "—"
    s = f"{r:.2f}".replace("-", "−").replace(".", ",")
    return s if abs(r) >= limiar(n) else f"({s})"


def tabela_md(longo: pd.DataFrame, linhas: list, colunas: list, valor: str = "pearson") -> str:
    r = longo.pivot(index="ativo", columns="coluna", values=valor)
    n = longo.pivot(index="ativo", columns="coluna", values="n")
    linhas = [l for l in linhas if l in r.index]
    colunas = [c for c in colunas if c in r.columns]
    out = ["| Ativo | " + " | ".join(colunas) + " |", "|---" * (len(colunas) + 1) + "|"]
    for l in linhas:
        out.append(f"| {l} | " + " | ".join(fmt_r(r.loc[l, c], n.loc[l, c]) for c in colunas) + " |")
    return "\n".join(out)


def salvar_heatmap(longo: pd.DataFrame, linhas: list, colunas: list, titulo: str,
                   arquivo: Path, vmax: float = None) -> None:
    r = longo.pivot(index="ativo", columns="coluna", values="pearson")
    n = longo.pivot(index="ativo", columns="coluna", values="n")
    linhas = [l for l in linhas if l in r.index]
    colunas = [c for c in colunas if c in r.columns]
    r, n = r.loc[linhas, colunas], n.loc[linhas, colunas]
    anot = np.vectorize(lambda v, k: "" if pd.isna(v) else
                        (f"{v:.2f}" if abs(v) >= limiar(k) else f"({v:.2f})"))(r.to_numpy(), n.to_numpy())
    vmax = vmax or max(0.1, float(np.nanmax(np.abs(r.to_numpy()))))
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(colunas) + 3), max(4, 0.45 * len(linhas) + 2)))
    sns.heatmap(r, annot=anot, fmt="", cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title(titulo + "\n(entre parenteses: |r| < 2/sqrt(n), insignificante)", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=130)
    plt.close(fig)
    logger.info("Heatmap salvo: %s", arquivo)


def alvos_bova11(bova: pd.DataFrame) -> pd.DataFrame:
    fech_ant = bova["Close"].shift(1)
    a = pd.DataFrame(index=bova.index)
    a["gap"] = bova["Open"] / fech_ant - 1
    a["abs_gap"] = a["gap"].abs()
    a["ret_cc"] = bova["Close"] / fech_ant - 1
    a["abs_ret_cc"] = a["ret_cc"].abs()
    a["ret_intraday"] = bova["Close"] / bova["Open"] - 1
    a["abs_intraday"] = a["ret_intraday"].abs()
    a["amplitude"] = (bova["High"] - bova["Low"]) / bova["Open"]
    # definicao usada na Etapa 1 (mantida para comparacao com a primeira rodada)
    a["amplitude_fech_ant"] = (bova["High"] - bova["Low"]) / fech_ant
    return a


def ultimo_antes(serie: pd.Series, datas: pd.DatetimeIndex) -> pd.Series:
    """Ultimo valor com data estritamente anterior a cada data (equivale a ffill + shift)."""
    serie = serie.dropna().sort_index()
    pos = serie.index.searchsorted(datas, side="left") - 1
    valores = np.where(pos >= 0, serie.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(valores, index=datas)


def features_proprias(dados: dict) -> pd.DataFrame:
    """Features conhecidas antes da abertura de t: nivel do VIX em t-1 e HAR da amplitude."""
    bova = dados[ALVO]
    amp = (bova["High"] - bova["Low"]) / bova["Open"]
    f = pd.DataFrame(index=bova.index)
    f["vix_nivel_t1"] = ultimo_antes(dados["VIX"]["Close"], bova.index)
    for j in JANELAS_HAR:
        f[f"amp_media_{j}d"] = amp.rolling(j).mean().shift(1)
    return f


# --------------------------------------------------------------------------- #
# Etapa 1 - Correlacoes diarias (5 anos)
# --------------------------------------------------------------------------- #
COLS_E1 = ["ret x ret BOVA (t)", "ret x ret BOVA (t-1)",
           "|ret| x |ret| BOVA (t)", "|ret| x |ret| BOVA (t-1)",
           "|ret| x amp BOVA (t)", "|ret| x amp BOVA (t-1)"]


def etapa1_diaria(anos: int = 5) -> dict:
    logger.info("Etapa 1: baixando %d anos de dados diarios...", anos)
    dados = {}
    for nome, ticker in ATIVOS_DIARIOS.items():
        df = diario(ticker, period=f"{anos}y", auto_adjust=True)
        if not df.empty:
            dados[nome] = df
            logger.info("  %-12s %-10s %4d dias (%s a %s)%s", nome, ticker, len(df),
                        df.index[0].date(), df.index[-1].date(),
                        " [datas recuadas 1 barra]" if eh_cambio(ticker) else "")

    bova = dados[ALVO]
    pregoes = bova.index
    # Fechamentos alinhados aos pregoes do BOVA11: ultimo preco conhecido (ffill)
    fechamentos = pd.DataFrame({
        nome: df["Close"].reindex(pregoes, method="ffill") for nome, df in dados.items()
    })
    retornos = fechamentos.pct_change(fill_method=None)
    alvos = alvos_bova11(bova).rename(columns={"ret_cc": "retorno", "abs_ret_cc": "abs_retorno"})

    linhas = []
    for nome in retornos.columns:
        for transf, serie in (("retorno", retornos[nome]), ("abs_retorno", retornos[nome].abs())):
            for lag in (0, 1):
                if nome == ALVO and lag == 0:
                    continue
                for alvo in ("retorno", "abs_retorno", "amplitude_fech_ant"):
                    linhas.append({"ativo": nome, "ticker": ATIVOS_DIARIOS[nome],
                                   "transformacao": transf, "defasagem": lag, "alvo_bova11": alvo,
                                   **correlacionar(serie.shift(lag), alvos[alvo])})
    corr = pd.DataFrame(linhas)
    corr["coluna"] = (corr["transformacao"].map({"retorno": "ret", "abs_retorno": "|ret|"})
                      + " x " + corr["alvo_bova11"].map({"retorno": "ret BOVA", "abs_retorno": "|ret| BOVA",
                                                         "amplitude_fech_ant": "amp BOVA"})
                      + corr["defasagem"].map({0: " (t)", 1: " (t-1)"}))
    matriz = retornos.corr()
    n_matriz = retornos.notna().astype(int).T @ retornos.notna().astype(int)

    corr.drop(columns="coluna").to_csv(OUT_DIR / "correlacao_diaria_bova11.csv", index=False, float_format="%.6f")
    matriz.to_csv(OUT_DIR / "matriz_correlacao_retornos.csv", float_format="%.6f")
    salvar_heatmap(corr, list(ATIVOS_DIARIOS), COLS_E1, "Correlacao diaria com o BOVA11 (Pearson, 5 anos)",
                   OUT_DIR / "heatmap_correlacao_bova11.png")
    matriz_longa = matriz.stack().rename("pearson").reset_index()
    matriz_longa.columns = ["ativo", "coluna", "pearson"]
    matriz_longa["n"] = n_matriz.stack().to_numpy()
    salvar_heatmap(matriz_longa, list(matriz.index), list(matriz.columns),
                   "Matriz de correlacao de retornos diarios (5 anos)",
                   OUT_DIR / "heatmap_matriz_ativos.png", vmax=1)
    return {"corr": corr, "matriz": matriz, "matriz_longa": matriz_longa, "dados": dados}


# --------------------------------------------------------------------------- #
# Etapa 2 - Overnight (1h, prepost)
# --------------------------------------------------------------------------- #
def candles(h: pd.DataFrame, duracao: pd.Timedelta) -> pd.DataFrame:
    inicio = pd.DatetimeIndex(h.index).tz_convert(TZ)
    return pd.DataFrame({"fim": (inicio + duracao).as_unit("ns"),
                         "preco": h["Close"].to_numpy()}).dropna().sort_values("fim")


def preco_em(c: pd.DataFrame, refs: pd.Series) -> pd.DataFrame:
    """Close do ultimo candle encerrado ate cada horario de referencia (ffill).

    refs: Series indexada por pregao com o timestamp (tz-aware) de referencia.
    """
    alvo = pd.DataFrame({"pregao": refs.index, "ref": pd.DatetimeIndex(refs).as_unit("ns")}).sort_values("ref")
    m = pd.merge_asof(alvo, c, left_on="ref", right_on="fim", direction="backward")
    m["idade_h"] = (m["ref"] - m["fim"]).dt.total_seconds() / 3600
    return m.set_index("pregao")[["preco", "idade_h"]].reindex(refs.index)


def horario(pregoes: pd.DatetimeIndex, hora: pd.Timedelta) -> pd.Series:
    return pd.Series(pregoes.tz_localize(TZ) + hora, index=pregoes)


def fechamento_b3_anterior(todos_pregoes: pd.DatetimeIndex, pregoes: pd.DatetimeIndex,
                           inicio: pd.Timestamp) -> pd.Series:
    """Horario (BRT) do fechamento da B3 no pregao anterior a cada pregao.

    Vem do fim do ultimo candle de 1h do BOVA11 em D-1; sem candle, usa 17h.
    """
    h = baixar("BOVA11.SA", start=inicio, interval="1h", auto_adjust=False)
    ini = pd.DatetimeIndex(h.index).tz_convert(TZ)
    fim_dia = pd.Series(ini + pd.Timedelta(hours=1), index=para_datas(ini)).groupby(level=0).max()
    anteriores = todos_pregoes[todos_pregoes.get_indexer(pregoes) - 1]
    padrao = pd.Series(anteriores.tz_localize(TZ) + FECHAMENTO_B3_PADRAO, index=anteriores)
    fech = fim_dia.reindex(anteriores).fillna(padrao)
    return pd.Series(fech.to_numpy(), index=pregoes)


COLS_E2 = ["var x gap", "var x ret fech.", "var x ret intrad.",
           "|var| x |gap|", "|var| x |ret fech.|", "|var| x |close-open|/open", "|var| x amplitude"]
PARES_E2 = [("var_overnight", "gap"), ("var_overnight", "ret_cc"), ("var_overnight", "ret_intraday"),
            ("abs_var_overnight", "abs_gap"), ("abs_var_overnight", "abs_ret_cc"),
            ("abs_var_overnight", "abs_intraday"), ("abs_var_overnight", "amplitude")]


def rotulo(transf: str, alvo: str) -> str:
    return ("var" if transf == "var_overnight" else "|var|") + " x " + ROTULOS_ALVO[alvo]


def etapa2_overnight(dados: dict) -> dict:
    bova = dados[ALVO]
    inicio = HOJE - pd.Timedelta(days=DIAS_INTRADAY)
    # primeira semana fica de fora para o pregao anterior tambem ter candles de 1h
    pregoes = bova.index[bova.index >= inicio + pd.Timedelta(days=7)]
    alvos = alvos_bova11(bova).reindex(pregoes)
    ref_9h = horario(pregoes, HORA_9H)
    ref_fech_b3 = fechamento_b3_anterior(bova.index, pregoes, inicio)
    logger.info("Etapa 2: %d pregoes do BOVA11 desde %s", len(pregoes), inicio.date())

    variacoes, diagnostico = {}, []
    for nome, ticker in ATIVOS_OVERNIGHT.items():
        h1 = baixar(ticker, start=inicio, interval="1h", prepost=True, auto_adjust=False)
        if h1.empty:
            continue
        c = candles(h1, pd.Timedelta(hours=1))
        p9 = preco_em(c, ref_9h)
        if eh_cambio(ticker):
            pf = preco_em(c, ref_fech_b3)
            fech_ant, origem, idade_fech = pf["preco"], "1h no fechamento B3 de D-1", pf["idade_h"].median()
        else:
            d1 = diario(ticker, start=inicio - pd.Timedelta(days=10), auto_adjust=False)
            fech_ant, origem, idade_fech = ultimo_antes(d1["Close"], pregoes), "fechamento diario anterior", np.nan
        var = p9["preco"] / fech_ant - 1
        frac = float((p9["idade_h"].dropna() <= IDADE_MAX_FRESCO_H).mean())
        diagnostico.append({
            "ativo": nome, "ticker": ticker, "fechamento_anterior": origem,
            "idade_mediana_fech_ant_h": idade_fech,
            "pregoes_com_preco": int(var.notna().sum()),
            "idade_mediana_9h_h": float(p9["idade_h"].median()),
            "fracao_preco_fresco": frac,
            "desvio_var_overnight": float(var.std()),
            "usado": frac >= FRACAO_MIN_FRESCO,
        })
        variacoes[nome] = var
        logger.info("  %-12s %-10s idade 9h %.1fh | fresco %.0f%% | %s", nome, ticker,
                    p9["idade_h"].median(), 100 * frac, origem)
        if nome == "EWZ":
            idade_ewz_1h = p9["idade_h"]

    diag = pd.DataFrame(diagnostico)
    var_df = pd.DataFrame(variacoes)
    usados = diag.loc[diag["usado"], "ativo"].tolist()

    linhas = []
    for nome in usados:
        series = {"var_overnight": var_df[nome], "abs_var_overnight": var_df[nome].abs()}
        for transf, alvo in PARES_E2:
            linhas.append({"ativo": nome, "ticker": ATIVOS_OVERNIGHT[nome], "transformacao": transf,
                           "alvo_bova11": alvo, "coluna": rotulo(transf, alvo),
                           **correlacionar(series[transf], alvos[alvo])})
    corr = pd.DataFrame(linhas)

    # Features proprias (VIX nivel t-1, HAR) no mesmo periodo e em 5 anos
    feats = features_proprias(dados)
    alvos_5a = alvos_bova11(bova)
    linhas = []
    for janela, idx in (("~2 anos", pregoes), ("5 anos", bova.index)):
        for f in feats.columns:
            for alvo in ("abs_gap", "abs_ret_cc", "abs_intraday", "amplitude"):
                linhas.append({"ativo": f, "janela": janela, "alvo_bova11": alvo,
                               "coluna": f"{ROTULOS_ALVO[alvo]} ({janela})",
                               **correlacionar(feats[f].reindex(idx), alvos_5a[alvo].reindex(idx))})
    corr_feats = pd.DataFrame(linhas)

    ewz = analise_ewz_5m(dados, alvos_5a)
    parcial = correlacoes_parciais(var_df, feats.reindex(pregoes), alvos)

    pd.concat([var_df.add_prefix("var_on_"), feats.reindex(pregoes),
               idade_ewz_1h.rename("ewz_idade_9h_h"), alvos.add_prefix("bova11_")], axis=1) \
        .to_csv(OUT_DIR / "overnight_dados.csv", index_label="pregao", float_format="%.6f")
    corr.drop(columns="coluna").to_csv(OUT_DIR / "correlacao_overnight_bova11.csv", index=False, float_format="%.6f")
    corr_feats.drop(columns="coluna").to_csv(OUT_DIR / "correlacao_features_proprias.csv", index=False,
                                             float_format="%.6f")
    parcial.to_csv(OUT_DIR / "correlacao_parcial_cambio.csv", index=False, float_format="%.6f")
    diag.to_csv(OUT_DIR / "overnight_diagnostico.csv", index=False, float_format="%.4f")

    salvar_heatmap(corr, list(ATIVOS_OVERNIGHT), COLS_E2,
                   "Variacao overnight ate 9h BRT x BOVA11 (Pearson, ~2 anos)",
                   OUT_DIR / "heatmap_overnight_bova11.png")
    cols_feats = list(dict.fromkeys(corr_feats["coluna"]))
    salvar_heatmap(corr_feats, list(feats.columns), cols_feats,
                   "Features proprias (VIX nivel t-1, amplitude media) x BOVA11",
                   OUT_DIR / "heatmap_features_proprias.png")
    return {"corr": corr, "diag": diag, "var": var_df, "corr_feats": corr_feats,
            "cols_feats": cols_feats, "parcial": parcial, "ewz": ewz, "idade_ewz_1h": idade_ewz_1h}


def analise_ewz_5m(dados: dict, alvos: pd.DataFrame) -> dict:
    """EWZ em candles de 5m: horario do ultimo negocio antes das 9h e preco 9h x 9h15.

    No pre-mercado o Yahoo informa volume 0, mas so cria o candle de 5m quando ha
    negocio; por isso o fim do ultimo candle e usado como horario do ultimo negocio.
    """
    h5 = baixar("EWZ", period=f"{DIAS_5M + 1}d", interval="5m", prepost=True, auto_adjust=False)
    c = candles(h5, pd.Timedelta(minutes=5))
    primeiro_dia = c["fim"].min().tz_convert(TZ).tz_localize(None).normalize() + pd.Timedelta(days=1)
    pregoes = dados[ALVO].index[dados[ALVO].index >= primeiro_dia]
    fech_ant = ultimo_antes(dados["EWZ"]["Close"], pregoes)
    p9, p915 = preco_em(c, horario(pregoes, HORA_9H)), preco_em(c, horario(pregoes, HORA_9H15))
    df = pd.DataFrame({
        "idade_9h_min": p9["idade_h"] * 60,
        "idade_9h15_min": p915["idade_h"] * 60,
        "var_9h": p9["preco"] / fech_ant - 1,
        "var_9h15": p915["preco"] / fech_ant - 1,
    }, index=pregoes)
    df["dif_9h15_9h_bps"] = (p915["preco"] / p9["preco"] - 1) * 1e4
    df = df.join(alvos.reindex(pregoes).add_prefix("bova11_"))
    df.to_csv(OUT_DIR / "ewz_5m_9h_vs_9h15.csv", index_label="pregao", float_format="%.6f")

    linhas = []
    for col in ("var_9h", "var_9h15"):
        for transf, alvo in PARES_E2:
            x = df[col] if transf == "var_overnight" else df[col].abs()
            linhas.append({"ativo": "EWZ " + col.replace("var_", ""), "coluna": rotulo(transf, alvo),
                           **correlacionar(x, df["bova11_" + alvo])})
    resumo = {
        "pregoes": len(df),
        "periodo": f"{pregoes[0].date()} a {pregoes[-1].date()}",
        "idade_9h_mediana_min": float(df["idade_9h_min"].median()),
        "idade_9h_p90_min": float(df["idade_9h_min"].quantile(0.9)),
        "frac_ultimo_negocio_15min": float((df["idade_9h_min"] <= 15).mean()),
        "frac_ultimo_negocio_60min": float((df["idade_9h_min"] <= 60).mean()),
        "idade_9h15_mediana_min": float(df["idade_9h15_min"].median()),
        "corr_var_9h_x_9h15": float(df[["var_9h", "var_9h15"]].corr().iloc[0, 1]),
        "dif_abs_mediana_bps": float(df["dif_9h15_9h_bps"].abs().median()),
        "desvio_var_9h": float(df["var_9h"].std()),
        "desvio_var_9h15": float(df["var_9h15"].std()),
    }
    return {"dados": df, "corr": pd.DataFrame(linhas), "resumo": resumo}


def correlacoes_parciais(var_df: pd.DataFrame, feats: pd.DataFrame, alvos: pd.DataFrame) -> pd.DataFrame:
    """USDBRL e DXY acrescentam algo alem do EWZ pre-mercado?"""
    linhas = []
    for moeda in ("USDBRL", "DXY"):
        for transf, alvo in [("var_overnight", "gap"), ("abs_var_overnight", "abs_gap"),
                             ("abs_var_overnight", "abs_intraday"), ("abs_var_overnight", "amplitude")]:
            f = (lambda s: s) if transf == "var_overnight" else (lambda s: s.abs())
            x, y = f(var_df[moeda]), alvos[alvo]
            controles = {
                "nenhum (simples)": pd.DataFrame(index=var_df.index),
                "EWZ": pd.DataFrame({"ewz": f(var_df["EWZ"])}),
                "EWZ + ES=F + amp_media_22d": pd.DataFrame({"ewz": f(var_df["EWZ"]), "es": f(var_df["ES_FUT"]),
                                                            "har22": feats["amp_media_22d"]}),
            }
            for nome_ctrl, ctrl in controles.items():
                res = correlacionar(x, y) if ctrl.empty else correlacao_parcial(x, y, ctrl)
                linhas.append({"ativo": moeda, "transformacao": transf, "alvo_bova11": alvo,
                               "coluna": rotulo(transf, alvo), "controle": nome_ctrl,
                               **{k: res[k] for k in ("pearson", "p_valor", "n", "significativo")}})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r1 = etapa1_diaria()
    r2 = etapa2_overnight(r1["dados"])

    print("\n### ETAPA 1: correlacao diaria com o BOVA11 (5 anos)\n")
    print(tabela_md(r1["corr"], list(ATIVOS_DIARIOS), COLS_E1))
    cambio = ["USDBRL", "DXY", "EURBRL", "GBPBRL", "USDJPY", "EURUSD"]
    print("\n### ETAPA 1: matriz (recorte cambio x bolsa)\n")
    print(tabela_md(r1["matriz_longa"], cambio, ["BOVA11", "EWZ", "SP500", "VIX", "DXY", "EURUSD"]))
    print("\n### ETAPA 2: diagnostico\n")
    print(r2["diag"].round(3).to_string(index=False))
    print("\n### ETAPA 2: overnight ate 9h BRT x BOVA11\n")
    print(tabela_md(r2["corr"], list(ATIVOS_OVERNIGHT), COLS_E2))
    print("\n### ETAPA 2: overnight ate 9h BRT x BOVA11 (Spearman, robusto a outliers)\n")
    print(tabela_md(r2["corr"], list(ATIVOS_OVERNIGHT), COLS_E2, valor="spearman"))
    print("\n### Features proprias\n")
    linhas_feats = ["vix_nivel_t1"] + [f"amp_media_{j}d" for j in JANELAS_HAR]
    print(tabela_md(r2["corr_feats"], linhas_feats, r2["cols_feats"]))
    print("\n(Spearman)\n")
    print(tabela_md(r2["corr_feats"], linhas_feats, r2["cols_feats"], valor="spearman"))
    print("\n### EWZ 5m: 9h x 9h15\n")
    print(r2["ewz"]["resumo"])
    print(tabela_md(r2["ewz"]["corr"], ["EWZ 9h", "EWZ 9h15"], COLS_E2))
    print("\n(Spearman)\n")
    print(tabela_md(r2["ewz"]["corr"], ["EWZ 9h", "EWZ 9h15"], COLS_E2, valor="spearman"))
    print("\n### EWZ 1h: idade do ultimo candle antes das 9h (h), distribuicao\n")
    print(r2["idade_ewz_1h"].value_counts(normalize=True).sort_index().head(8).round(3).to_string())
    print("\n### Correlacao parcial do cambio\n")
    p = r2["parcial"].copy()
    p["r"] = [fmt_r(r, n) for r, n in zip(p["pearson"], p["n"])]
    print(p.pivot_table(index=["ativo", "coluna"], columns="controle", values="r", aggfunc="first").to_string())
    print(p.pivot_table(index=["ativo", "coluna"], columns="controle", values="p_valor").round(4).to_string())
    print(f"\nArquivos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
