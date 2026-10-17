import sys; sys.path.insert(0, "src")
import pandas as pd
from correlacao_ativos import diario
tk = {"BOVA11":"BOVA11.SA","DXY":"DX-Y.NYB","USDBRL":"BRL=X","EURUSD":"EURUSD=X","USDJPY":"JPY=X","EURBRL":"EURBRL=X"}
d = {k: diario(v, period="5y", auto_adjust=True)["Close"] for k, v in tk.items()}
dias = d["BOVA11"].index
for desloc in (0, 1):
    f = {}
    for k, s in d.items():
        if k.startswith(("USD","EUR")) and desloc:
            s = s.copy(); s.index = s.index - pd.Timedelta(days=1)   # close datado D = fim do dia D-1
        f[k] = s.reindex(dias, method="ffill")
    r = pd.DataFrame(f).pct_change(fill_method=None)
    c = r.corr()
    print(f"FX deslocado {desloc}d: BOVA x USDBRL {c.loc['BOVA11','USDBRL']:+.2f} | BOVA x EURBRL {c.loc['BOVA11','EURBRL']:+.2f} | DXY x EURUSD {c.loc['DXY','EURUSD']:+.2f} | DXY x USDJPY {c.loc['DXY','USDJPY']:+.2f}")
