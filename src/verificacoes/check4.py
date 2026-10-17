import sys; sys.path.insert(0, "src")
import pandas as pd, numpy as np, yfinance as yf
from zoneinfo import ZoneInfo
BRT = ZoneInfo("America/Sao_Paulo")
from correlacao_ativos import baixar, diario
# 1) dias da semana das barras diarias de FX
d = diario("JPY=X", period="1y", auto_adjust=True)
print("JPY=X dias da semana:", d.index.dayofweek.value_counts().sort_index().to_dict(), "ultima:", d.index[-1].date())
# 2) horario dos candles 1h do BOVA11 (ultimo candle de cada dia, por mes)
h = baixar("BOVA11.SA", period="729d", interval="1h", auto_adjust=False)
idx = pd.DatetimeIndex(h.index).tz_convert(BRT)
ult = pd.Series(idx, index=idx.date).groupby(level=0).max()
prim = pd.Series(idx, index=idx.date).groupby(level=0).min()
df = pd.DataFrame({"prim": prim.dt.strftime("%H:%M"), "ult": ult.dt.strftime("%H:%M")})
df["mes"] = pd.to_datetime(df.index).to_period("M")
print(df.groupby("mes").agg(lambda s: s.value_counts().index[0]).tail(14).to_string())
# 3) EWZ pre-mercado: volume nos candles 1h e 5m
for itv, per in (("1h", "729d"), ("5m", "60d")):
    e = baixar("EWZ", period=per, interval=itv, prepost=True, auto_adjust=False)
    i = pd.DatetimeIndex(e.index).tz_convert(BRT)
    pre = e[(i.hour < 9) | ((i.hour == 9) & (i.minute < 15))]
    pre = pre[pd.DatetimeIndex(pre.index).tz_convert(BRT).hour >= 4]
    print(itv, "EWZ candles pre-9h15:", len(pre), "com volume>0:", int((pre["Volume"] > 0).sum()), "| inicio dados:", i[0])
