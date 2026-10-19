"""
dataset_9h15.py
===============
Fase C - Projeto IOSI. Montagem point-in-time dos datasets de treino.

Cada linha e um pregao t da B3 (calendario BVMF do pandas_market_calendars).
As features so usam o que era conhecido antes da hora de referencia de t (9h hoje;
9h15 quando houver snapshots suficientes). Os alvos sao todos do dia t.

Dataset "vol" (volatilidade / dia lateral), maior historico do Yahoo (desde 2008):
    har_1d, har_5d, har_22d   amplitude media do BOVA11 em 1, 5 e 22 pregoes ate t-1
    vix_nivel_t1              fechamento do VIX no ultimo dia (EUA) antes de t
    abs_ret_bova11_t1         |retorno de fechamento| do BOVA11 em t-1
    abs_ret_sp500_t1          |retorno| do S&P 500 no ultimo dia (EUA) antes de t

Dataset "gap" (candles de 1h com prepost, ~2 anos):
    var_on_<ativo>            preco as 9h BRT / ultimo fechamento diario antes de t - 1
    idade_<ativo>_h           horas entre o fim do ultimo candle e a referencia
    para EWZ, ES=F, VIX e BZ=F. Com hora="9h15", os precos vem dos snapshots gravados
    pelo coletor_snapshots.py (exige MIN_SNAPSHOTS dias).

Alvos: gap, ret_cc, ret_intraday, abs_intraday (|close-open|/open), amplitude e
lateral (abs_intraday < 1%).

Barra diaria do BOVA11: pregoes sem barra no Yahoo (ex.: 15/10/2026) sao montados com
os candles de 1h (fonte_barra = "agregado_1h"); sem candles, ficam "sem_barra" (alvos
NaN). Barras do Yahoo em dias sem pregao sao descartadas. Barras invalidas (volume 0,
high == low, open <= 0 ou open/close fora de [low, high]; 146 casos, ate 2021) sao
tratadas como ausentes: "agregado_1h" se houver candles, senao "barra_invalida".

Regras de disponibilidade (usadas tambem no teste point-in-time):
    - barra diaria com data D: conhecida a partir de D+1 (data < t);
    - candle intradiario: conhecido quando termina (fim <= referencia);
    - o calendario da B3 e conhecido de antemao.

Uso:
    python src/dataset_9h15.py            # grava data/datasets/*.csv e mostra o resumo
    python -m pytest tests                # teste point-in-time
"""

import logging
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TZ = ZoneInfo("America/Sao_Paulo")
HOJE = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)

DIARIOS = {
    "BOVA11":    "BOVA11.SA",
    "VIX":       "^VIX",
    "SP500":     "^GSPC",
    "EWZ":       "EWZ",
    "ES_FUT":    "ES=F",
    "BRENT_FUT": "BZ=F",
}
OVERNIGHT = {
    "EWZ":       "EWZ",
    "ES_FUT":    "ES=F",
    "VIX":       "^VIX",
    "BRENT_FUT": "BZ=F",
}
HORAS_REF = {"9h": pd.Timedelta(hours=9), "9h15": pd.Timedelta(hours=9, minutes=15),
             "10h30": pd.Timedelta(hours=10, minutes=30)}
JANELAS_HAR = (1, 5, 22)
LIMIAR_LATERAL = 0.01
DIAS_1H = 729                           # limite do Yahoo para 1h: 730 dias
MIN_SNAPSHOTS = 60

FEATS_VOL = [f"har_{j}d" for j in JANELAS_HAR] + ["vix_nivel_t1", "abs_ret_bova11_t1", "abs_ret_sp500_t1"]
FEATS_GAP = [f"var_on_{a}" for a in OVERNIGHT] + [f"idade_{a}_h" for a in OVERNIGHT]
ALVOS = ["gap", "ret_cc", "ret_intraday", "abs_intraday", "amplitude", "lateral"]

RAIZ = Path(__file__).parent.parent
DIR_SNAPSHOTS = RAIZ / "data" / "snapshots"
DIR_SAIDA = RAIZ / "data" / "datasets"


# --------------------------------------------------------------------------- #
# Dados brutos
# --------------------------------------------------------------------------- #
def baixar(ticker: str, **kwargs) -> pd.DataFrame:
    for tentativa in range(1, 4):
        try:
            df = yf.download(ticker, progress=False, multi_level_index=False, auto_adjust=False, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao baixar %s (tentativa %d): %s", ticker, tentativa, exc)
            df = pd.DataFrame()
        if not df.empty:
            return df.dropna(subset=["Close"])
    raise RuntimeError(f"Sem dados para {ticker} ({kwargs})")


def diario(ticker: str) -> pd.DataFrame:
    df = baixar(ticker, period="max", interval="1d")
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_convert(TZ).tz_localize(None) if idx.tz is not None else idx).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()[["Open", "High", "Low", "Close", "Volume"]]


def intradiario_1h(ticker: str, inicio: pd.Timestamp, prepost: bool) -> pd.DataFrame:
    h = baixar(ticker, start=inicio, interval="1h", prepost=prepost)
    ini = pd.DatetimeIndex(h.index).tz_convert(TZ).as_unit("ns")
    return pd.DataFrame({"inicio": ini, "fim": ini + pd.Timedelta(hours=1),
                         **{c: h[c].to_numpy() for c in ("Open", "High", "Low", "Close")}}) \
        .sort_values("fim").reset_index(drop=True)


def baixar_dados() -> dict:
    """Tudo o que os datasets usam, sem nenhum processamento que dependa da data t."""
    inicio_1h = HOJE - pd.Timedelta(days=DIAS_1H)
    d = {nome: diario(t) for nome, t in DIARIOS.items()}
    h1 = {"BOVA11": intradiario_1h("BOVA11.SA", inicio_1h, prepost=False)}
    h1.update({nome: intradiario_1h(t, inicio_1h, prepost=True) for nome, t in OVERNIGHT.items()})
    sessoes = mcal.get_calendar("BVMF").valid_days(d["BOVA11"].index[0], HOJE)
    sessoes = pd.DatetimeIndex(sessoes).tz_localize(None).normalize()
    return {"diario": d, "h1": h1, "sessoes": sessoes, "ate": HOJE, "ref_ate": None,
            "inicio_1h": inicio_1h}


def truncar(dados: dict, t: pd.Timestamp, hora: str = "9h") -> dict:
    """Dados brutos como eram conhecidos as `hora` do pregao t."""
    ref = t.tz_localize(TZ) + HORAS_REF[hora]
    return {
        "diario": {k: v[v.index < t] for k, v in dados["diario"].items()},
        "h1": {k: v[v["fim"] <= ref] for k, v in dados["h1"].items()},
        "sessoes": dados["sessoes"][dados["sessoes"] <= t],
        "ate": t, "ref_ate": ref, "inicio_1h": dados["inicio_1h"],
    }


# --------------------------------------------------------------------------- #
# Utilitarios
# --------------------------------------------------------------------------- #
def ultimo_antes(serie: pd.Series, datas: pd.DatetimeIndex) -> pd.Series:
    """Ultimo valor com data estritamente anterior a cada data (equivale a ffill + shift)."""
    serie = serie.dropna().sort_index()
    pos = serie.index.searchsorted(datas, side="left") - 1
    valores = np.where(pos >= 0, serie.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(valores, index=datas)


def preco_em(c: pd.DataFrame, refs: pd.Series) -> pd.DataFrame:
    """Close do ultimo candle encerrado ate cada referencia (tz-aware) e sua idade em horas."""
    alvo = pd.DataFrame({"pregao": refs.index, "ref": pd.DatetimeIndex(refs).as_unit("ns")}).sort_values("ref")
    m = pd.merge_asof(alvo, c[["fim", "Close"]].dropna(), left_on="ref", right_on="fim", direction="backward")
    m["idade_h"] = (m["ref"] - m["fim"]).dt.total_seconds() / 3600
    return m.set_index("pregao")[["Close", "idade_h"]].reindex(refs.index)


def barra_invalida(b: pd.DataFrame) -> pd.Series:
    """Barras sem negocio ou inconsistentes no historico do Yahoo."""
    tol = 1e-4
    return ((b["Volume"] <= 0) | (b["High"] <= b["Low"]) | (b["Open"] <= 0)
            | (b["Open"] > b["High"] * (1 + tol)) | (b["Open"] < b["Low"] * (1 - tol))
            | (b["Close"] > b["High"] * (1 + tol)) | (b["Close"] < b["Low"] * (1 - tol)))


def bova11_diario(dados: dict) -> pd.DataFrame:
    """OHLC do BOVA11 por pregao, com a origem de cada barra.

    Barras do Yahoo fora do calendario sao descartadas e barras invalidas viram NaN.
    Pregoes sem barra valida sao montados com os candles de 1h, so para dias anteriores
    a `ate` (dia ja encerrado).
    """
    sessoes = dados["sessoes"]
    b = dados["diario"]["BOVA11"].reindex(sessoes)
    invalida = b["Close"].notna() & barra_invalida(b)
    b.loc[invalida, ["Open", "High", "Low", "Close"]] = np.nan
    b["fonte_barra"] = np.select([invalida, b["Close"].notna()], ["barra_invalida", "yahoo_1d"], "sem_barra")

    h = dados["h1"]["BOVA11"]
    dia = h["inicio"].dt.tz_localize(None).dt.normalize()
    agg = h.assign(dia=dia).groupby("dia").agg(Open=("Open", "first"), High=("High", "max"),
                                               Low=("Low", "min"), Close=("Close", "last"))
    faltam = b.index[b["Close"].isna() & (b.index < dados["ate"]) & b.index.isin(agg.index)]
    b.loc[faltam, ["Open", "High", "Low", "Close"]] = agg.loc[faltam, ["Open", "High", "Low", "Close"]].to_numpy()
    b.loc[faltam, "fonte_barra"] = "agregado_1h"
    return b


def alvos(b: pd.DataFrame) -> pd.DataFrame:
    fech_ant = b["Close"].shift(1)
    a = pd.DataFrame(index=b.index)
    a["gap"] = b["Open"] / fech_ant - 1
    a["ret_cc"] = b["Close"] / fech_ant - 1
    a["ret_intraday"] = b["Close"] / b["Open"] - 1
    a["abs_intraday"] = a["ret_intraday"].abs()
    a["amplitude"] = (b["High"] - b["Low"]) / b["Open"]
    a["lateral"] = (a["abs_intraday"] < LIMIAR_LATERAL).astype(float).where(a["abs_intraday"].notna())
    a["fonte_barra"] = b["fonte_barra"]
    return a


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def montar_vol(dados: dict) -> pd.DataFrame:
    """Dataset de volatilidade / dia lateral: features diarias ate t-1."""
    b = bova11_diario(dados)
    amp = (b["High"] - b["Low"]) / b["Open"]
    f = pd.DataFrame(index=b.index)
    for j in JANELAS_HAR:
        f[f"har_{j}d"] = amp.rolling(j).mean().shift(1)
    f["vix_nivel_t1"] = ultimo_antes(dados["diario"]["VIX"]["Close"], b.index)
    f["abs_ret_bova11_t1"] = (b["Close"] / b["Close"].shift(1) - 1).abs().shift(1)
    sp = dados["diario"]["SP500"]["Close"]
    f["abs_ret_sp500_t1"] = ultimo_antes((sp / sp.shift(1) - 1).abs(), b.index)
    return f.join(alvos(b))


def montar_gap(dados: dict, hora: str = "9h") -> pd.DataFrame:
    """Dataset do gap: variacao overnight ate `hora` de t contra o ultimo fechamento diario."""
    b = bova11_diario(dados)
    sessoes = b.index[b.index >= dados["inicio_1h"] + pd.Timedelta(days=7)]
    if hora == "9h15":
        f = features_snapshots(sessoes)
    else:
        refs = pd.Series(sessoes.tz_localize(TZ) + HORAS_REF[hora], index=sessoes)
        f = pd.DataFrame(index=sessoes)
        for nome in OVERNIGHT:
            p = preco_em(dados["h1"][nome], refs)
            fech_ant = ultimo_antes(dados["diario"][nome]["Close"], sessoes)
            f[f"var_on_{nome}"] = p["Close"] / fech_ant - 1
            f[f"idade_{nome}_h"] = p["idade_h"]
    return f[FEATS_GAP].join(alvos(b).reindex(sessoes))


def features_snapshot_dia(arquivo: Path) -> dict:
    """Features overnight das 9h15 de um snapshot (data/snapshots/AAAA-MM-DD_9h15.csv).

    So usa candles de 5m encerrados ate 9h15 e o fechamento diario gravado no snapshot
    (data < t), entao o resultado e point-in-time mesmo se a coleta rodou atrasada.
    """
    t = pd.Timestamp(arquivo.name[:10])
    ref = t.tz_localize(TZ) + HORAS_REF["9h15"]
    s = pd.read_csv(arquivo)
    linha = {}
    for nome in OVERNIGHT:
        c = s[(s["ativo"] == nome) & (s["tipo"] == "5m")].copy()
        c["fim"] = pd.to_datetime(c["fim"], format="%Y-%m-%d %H:%M:%S%z").dt.tz_convert(TZ)
        c = c[c["fim"] <= ref]
        fech = s.loc[(s["ativo"] == nome) & (s["tipo"] == "1d"), "close"]
        if c.empty or fech.empty:
            continue
        linha[f"var_on_{nome}"] = c["close"].iloc[-1] / fech.iloc[-1] - 1
        linha[f"idade_{nome}_h"] = (ref - c["fim"].iloc[-1]).total_seconds() / 3600
    return linha


def features_snapshots(sessoes: pd.DatetimeIndex) -> pd.DataFrame:
    """Features overnight das 9h15 a partir de data/snapshots/AAAA-MM-DD_9h15.csv."""
    arquivos = sorted(DIR_SNAPSHOTS.glob("????-??-??_9h15.csv"))
    if len(arquivos) < MIN_SNAPSHOTS:
        raise RuntimeError(f"So ha {len(arquivos)} snapshots das 9h15 (minimo {MIN_SNAPSHOTS}). Use hora='9h'.")
    linhas = {pd.Timestamp(arq.name[:10]): features_snapshot_dia(arq) for arq in arquivos}
    return pd.DataFrame.from_dict(linhas, orient="index").reindex(index=sessoes, columns=FEATS_GAP)


# --------------------------------------------------------------------------- #
# Verificacao point-in-time (usada pelo teste)
# --------------------------------------------------------------------------- #
def verificar_point_in_time(montar, dados: dict, datas, colunas: list, hora: str = "9h") -> list:
    """Remonta cada linha t so com os dados conhecidos as `hora` de t e compara.

    Devolve a lista de divergencias (data, coluna, valor completo, valor truncado).
    """
    completo = montar(dados)
    divergencias = []
    for t in datas:
        truncado = montar(truncar(dados, t, hora))
        a = completo.loc[t, colunas].astype(float)
        b = truncado.loc[t, colunas].astype(float) if t in truncado.index else pd.Series(np.nan, index=colunas)
        for col in colunas:
            if not np.isclose(a[col], b[col], rtol=1e-12, atol=0, equal_nan=True):
                divergencias.append((t.date(), col, a[col], b[col]))
    return divergencias


# --------------------------------------------------------------------------- #
def resumo_lateral(df: pd.DataFrame, nome: str) -> None:
    lat = df["lateral"].dropna()
    print(f"\n### Proporcao de dias laterais (|close-open|/open < 1%) - {nome}\n")
    print(f"Total: {lat.mean():.1%} de {len(lat)} pregoes ({lat.index[0].date()} a {lat.index[-1].date()})")
    for anos in (10, 5, 2, 1):
        rec = lat[lat.index > HOJE - pd.DateOffset(years=anos)]
        print(f"Ultimos {anos:2d} anos: {rec.mean():.1%} (n = {len(rec)})")
    por_ano = lat.groupby(lat.index.year).agg(["mean", "size"])
    print("\nPor ano: " + " | ".join(f"{a}: {m:.0%}" for a, (m, _) in por_ano.iterrows()))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    dados = baixar_dados()

    vol = montar_vol(dados)
    gap = montar_gap(dados)
    vol.to_csv(DIR_SAIDA / "dataset_vol_lateral.csv", index_label="pregao", float_format="%.10f")
    gap.to_csv(DIR_SAIDA / "dataset_gap_9h.csv", index_label="pregao", float_format="%.10f")

    for nome, df, feats in (("vol/lateral", vol, FEATS_VOL), ("gap (9h)", gap, FEATS_GAP)):
        completas = df[feats + ALVOS].dropna()
        print(f"\n### Dataset {nome}: {len(df)} pregoes ({df.index[0].date()} a {df.index[-1].date()}), "
              f"{len(completas)} com features e alvos completos")
        print("fonte_barra:", df["fonte_barra"].value_counts().to_dict())
        print("NaN por coluna:", {c: int(n) for c, n in df[feats + ALVOS].isna().sum().items() if n})
        print(df[feats + ALVOS].describe().T[["mean", "std", "min", "max"]].round(4).to_string())

    print("\n### Pregoes com barra reconstruida ou ausente\n")
    fora = vol[vol["fonte_barra"] != "yahoo_1d"]
    print(fora.groupby([fora.index.year, "fonte_barra"]).size().unstack(fill_value=0).to_string())
    print(fora.loc[fora["fonte_barra"] == "agregado_1h", ["gap", "abs_intraday", "amplitude", "lateral"]]
          .round(4).to_string())
    descartadas = dados["diario"]["BOVA11"].index.difference(dados["sessoes"])
    print(f"\nBarras do Yahoo fora do calendario da B3 (descartadas): {len(descartadas)}")

    # Qualidade da reconstrucao: candles de 1h agregados x barra diaria do Yahoo
    h = dados["h1"]["BOVA11"]
    agg = h.assign(dia=h["inicio"].dt.tz_localize(None).dt.normalize()).groupby("dia") \
        .agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
    d = dados["diario"]["BOVA11"][["Open", "High", "Low", "Close"]].reindex(agg.index).dropna()
    erro = ((agg.loc[d.index] / d - 1).abs() * 1e4)
    print("\n### Agregado de 1h x barra diaria (erro absoluto em bps, dias com as duas)\n")
    print(erro.describe(percentiles=[0.5, 0.9]).loc[["count", "50%", "90%", "max"]].round(1).to_string())
    lat_agg = alvos(agg.loc[d.index].assign(fonte_barra=""))["lateral"]
    lat_d = alvos(d.assign(fonte_barra=""))["lateral"]
    print(f"Concordancia do rotulo 'lateral' (agregado x diario): {(lat_agg == lat_d).mean():.1%}")

    resumo_lateral(vol, "dataset vol/lateral")
    print(f"\nArquivos em: {DIR_SAIDA}")


if __name__ == "__main__":
    main()
