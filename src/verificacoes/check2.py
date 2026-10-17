import sys; sys.path.insert(0, "src")
import pandas as pd, numpy as np
from correlacao_ativos import baixar, diario
ini = pd.Timestamp.now().normalize() - pd.Timedelta(days=300)
for t in ["BRL=X", "EURBRL=X", "JPY=X", "ES=F"]:
    h = baixar(t, start=ini, interval="1h", prepost=True, auto_adjust=False)
    d = diario(t, start=ini, auto_adjust=False).iloc[:-1]
    fim = pd.DatetimeIndex(h.index).tz_convert("America/Sao_Paulo") + pd.Timedelta(hours=1)
    s = pd.Series(h["Close"].to_numpy(), index=fim)
    res = {}
    for off in range(-6, 31, 1):
        ts = (d.index.tz_localize("America/Sao_Paulo") + pd.Timedelta(hours=off))
        p = s.reindex(ts)  # so candles que terminam exatamente nesse horario
        e = (p.to_numpy() / d["Close"].to_numpy() - 1)
        e = e[~np.isnan(e)]
        if len(e) > 30: res[off] = (np.median(np.abs(e))*1e4, len(e))
    best = sorted(res.items(), key=lambda kv: kv[1][0])[:4]
    print(t, " melhores horarios (h apos 00h BRT da data -> erro mediano bps, n):", [(k, round(v[0],2), v[1]) for k, v in best])
