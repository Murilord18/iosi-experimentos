import sys; sys.path.insert(0, "src")
import pandas as pd
from zoneinfo import ZoneInfo
BRT = ZoneInfo("America/Sao_Paulo")
from correlacao_ativos import baixar
e = baixar("EWZ", period="60d", interval="5m", prepost=True, auto_adjust=False)
e.index = pd.DatetimeIndex(e.index).tz_convert(BRT)
for dia in sorted(set(e.index.date))[-3:]:
    x = e[(e.index.date == dia) & (e.index.hour >= 5) & (e.index.hour < 10)][["Open","High","Low","Close","Volume"]]
    print(dia, len(x), "candles; com High!=Low:", int((x.High != x.Low).sum()))
    print(x.iloc[::4].round(3).to_string())
