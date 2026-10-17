import sys; sys.path.insert(0, "src")
import pandas as pd, numpy as np, yfinance as yf
from correlacao_ativos import baixar, diario
ini = pd.Timestamp.now().normalize() - pd.Timedelta(days=200)
for t in ["BRL=X", "ES=F", "EWZ", "^VIX", "DX-Y.NYB"]:
    h = baixar(t, start=ini, interval="1h", prepost=True, auto_adjust=False)
    d = diario(t, start=ini, auto_adjust=False)
    idx = pd.DatetimeIndex(h.index).tz_convert("America/Sao_Paulo")
    s = pd.Series(h["Close"].to_numpy(), index=idx)
    # para cada data diaria, achar o horario BRT do candle de 1h cujo close mais se aproxima do close diario
    hrs = []
    for dt, c in d["Close"].iloc[-60:].items():
        janela = s[(s.index >= pd.Timestamp(dt).tz_localize("America/Sao_Paulo") - pd.Timedelta(hours=12)) & (s.index < pd.Timestamp(dt).tz_localize("America/Sao_Paulo") + pd.Timedelta(hours=36))]
        if janela.empty: continue
        k = (janela / c - 1).abs().idxmin()
        hrs.append(((k - pd.Timestamp(dt).tz_localize("America/Sao_Paulo")).total_seconds()/3600 + 1, (janela/c-1).abs().min()))
    a = np.array(hrs)
    print(f"{t:10s} fim do candle mais proximo do close diario (h apos 00h BRT da data): mediana {np.median(a[:,0]):.0f}  | erro mediano {np.median(a[:,1])*1e4:.1f} bps")
