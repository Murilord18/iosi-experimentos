"""
leitura_1030.py
===============
Projeto IOSI - leitura point-in-time do pregao as 10h30 (BRT), com a B3 ja aberta.

Funcoes comuns ao predictor_1030.py, ao pesquisa_1030.py e ao teste point-in-time:
so entram candles de 5m ja ENCERRADOS ate a referencia (fim <= 10h30 de t), mesmo que
o arquivo ou o download traga candles posteriores.

Candles: DataFrame com inicio e fim (tz-aware, BRT) e open, high, low, close, volume
(formato do snapshot do coletor_snapshots.py; de_yahoo() converte o download do Yahoo).

Observacoes sobre o BOVA11 no Yahoo (5m):
    - o horario do candle e o de INICIO; o primeiro candle do dia costuma ser o das 10h05
      (o leilao de abertura do ETF termina por volta de 10h08), e o open dele e a abertura
      do dia;
    - os dados da B3 chegam com ~15 min de atraso, por isso o coletor espera o candle
      que termina as 10h30 (coletor_snapshots.py, modo manha_1030).
"""

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TZ = ZoneInfo("America/Sao_Paulo")
ABERTURA_B3 = pd.Timedelta(hours=10)
REF_1030 = pd.Timedelta(hours=10, minutes=30)
DURACAO_5M = pd.Timedelta(minutes=5)
EXTERNOS = {"EWZ": "EWZ", "ES_FUT": "ES=F", "VIX": "^VIX"}     # futuro do S&P, nao o indice a vista


def referencia(t: pd.Timestamp, hora: pd.Timedelta = REF_1030) -> pd.Timestamp:
    """10h30 BRT do pregao t (t sem fuso)."""
    return pd.Timestamp(t).normalize().tz_localize(TZ) + hora


def de_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    """Download de 5m do Yahoo (indice = inicio do candle) -> formato de candles."""
    if df.empty:
        return pd.DataFrame(columns=["inicio", "fim", "open", "high", "low", "close", "volume"])
    ini = pd.DatetimeIndex(df.index).tz_convert(TZ)
    return pd.DataFrame({"inicio": ini, "fim": ini + DURACAO_5M,
                         **{c.lower(): df[c].to_numpy() for c in ("Open", "High", "Low", "Close", "Volume")}}) \
        .sort_values("inicio").reset_index(drop=True)


def ler_snapshot(arquivo: Path) -> tuple[dict, dict]:
    """Snapshot do coletor -> ({ativo: candles de 5m}, {ativo: (data, fechamento diario)})."""
    s = pd.read_csv(arquivo, dtype={"inicio": str, "fim": str})
    candles, fechamentos = {}, {}
    for ativo, g in s.groupby("ativo"):
        c = g[g["tipo"] == "5m"].copy()
        if len(c):
            for col in ("inicio", "fim"):
                c[col] = pd.to_datetime(c[col], format="%Y-%m-%d %H:%M:%S%z").dt.tz_convert(TZ)
            candles[ativo] = c[["inicio", "fim", "open", "high", "low", "close", "volume"]] \
                .sort_values("inicio").reset_index(drop=True)
        d = g[g["tipo"] == "1d"].sort_values("inicio")
        if len(d):
            fechamentos[ativo] = (pd.Timestamp(d["inicio"].iloc[-1]), float(d["close"].iloc[-1]))
    return candles, fechamentos


def encerrados(c: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    """So os candles ja encerrados na referencia (fim <= ref)."""
    return c[c["fim"] <= ref]


def pregao_ate(c: pd.DataFrame, t: pd.Timestamp, ref: pd.Timestamp = None) -> pd.DataFrame:
    """Candles do BOVA11 no pregao t, da abertura (10h) ate a referencia."""
    ref = ref if ref is not None else referencia(t)
    c = encerrados(c, ref)
    return c[c["inicio"] >= referencia(t, ABERTURA_B3)]


def leitura_bova(c: pd.DataFrame, t: pd.Timestamp, ref: pd.Timestamp = None) -> dict:
    """Abertura real, preco na referencia e extremos ja ocorridos (so candles com fim <= ref)."""
    ref = ref if ref is not None else referencia(t)
    p = pregao_ate(c, t, ref)
    if p.empty:
        return {}
    abertura, preco = float(p["open"].iloc[0]), float(p["close"].iloc[-1])
    alta, baixa = float(p["high"].max()), float(p["low"].min())
    return {
        "abertura_real": abertura, "preco_1030": preco,
        "preco_1030_fim": p["fim"].iloc[-1], "idade_preco_1030_min": (ref - p["fim"].iloc[-1]).total_seconds() / 60,
        "high_ate_1030": alta, "low_ate_1030": baixa, "volume_ate_1030": float(p["volume"].sum()),
        "n_candles_ate_1030": len(p),
        "amp_ate_1030": (alta - baixa) / abertura, "ret_ate_1030": preco / abertura - 1,
    }


def variacao_ate(c: pd.DataFrame, fech_anterior: float, ref: pd.Timestamp) -> dict:
    """Ultimo preco encerrado ate a referencia / fechamento diario anterior - 1, e sua idade (h)."""
    c = encerrados(c, ref)
    if c.empty or not np.isfinite(fech_anterior):
        return {"var": np.nan, "idade_h": np.nan}
    return {"var": float(c["close"].iloc[-1]) / fech_anterior - 1,
            "idade_h": (ref - c["fim"].iloc[-1]).total_seconds() / 3600}


def features_1030(candles: dict, fech_ant: dict, t: pd.Timestamp) -> dict:
    """Features das 10h30 de t.

    candles: {ativo: candles de 5m} (BOVA11 e EXTERNOS); fech_ant: {ativo: fechamento
    diario anterior a t}. Nenhum candle com fim > 10h30 de t e usado.
    """
    ref = referencia(t)
    f = {}
    lb = leitura_bova(candles.get("BOVA11", pd.DataFrame(columns=["inicio", "fim"])), t, ref)
    if lb:
        f.update({k: lb[k] for k in ("abertura_real", "preco_1030", "high_ate_1030", "low_ate_1030",
                                     "volume_ate_1030", "amp_ate_1030", "ret_ate_1030")})
        f["gap_real"] = lb["abertura_real"] / fech_ant["BOVA11"] - 1 if "BOVA11" in fech_ant else np.nan
    for nome in EXTERNOS:
        if nome in candles and nome in fech_ant:
            v = variacao_ate(candles[nome], fech_ant[nome], ref)
            f[f"var_1030_{nome}"], f[f"idade_1030_{nome}_h"] = v["var"], v["idade_h"]
    return f
