"""Previa rapida da Fase D: walk-forward mensal (janela expansiva) nos datasets da Fase C."""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import spearmanr, pearsonr

base = sys.argv[1]
vol = pd.read_csv(f"{base}/data/datasets/dataset_vol_lateral.csv", parse_dates=["pregao"])
gap = pd.read_csv(f"{base}/data/datasets/dataset_gap_9h.csv", parse_dates=["pregao"])


def walk_forward(df, feats, target, models, min_train, start=None):
    df = df.dropna(subset=feats + [target]).sort_values("pregao").reset_index(drop=True)
    df["mes"] = df["pregao"].dt.to_period("M")
    meses = df["mes"].unique()
    preds = {k: np.full(len(df), np.nan) for k in models}
    for m in meses:
        tr = df["mes"] < m
        te = df["mes"] == m
        if tr.sum() < min_train or (start and str(m) < start):
            continue
        for k, mk in models.items():
            preds[k][te.values] = mk(df.loc[tr], df.loc[te], feats, target)
    mask = ~np.isnan(next(iter(preds.values())))
    out = df.loc[mask, ["pregao", target]].copy()
    for k in preds:
        out[k] = preds[k][mask]
    return out


# ---------- modelos ----------
def media(tr, te, f, y):
    return np.full(len(te), tr[y].mean())

def zero(tr, te, f, y):
    return np.zeros(len(te))

def fit_pred(est, cols=None):
    def _f(tr, te, f, y):
        c = cols or f
        e = est()
        e.fit(tr[c], tr[y])
        return e.predict(te[c])
    return _f

def fit_proba(est, cols=None):
    def _f(tr, te, f, y):
        c = cols or f
        e = est()
        e.fit(tr[c], tr[y].astype(int))
        return e.predict_proba(te[c])[:, 1]
    return _f

def log_fit(est, cols=None):  # modelo em log(amplitude), volta para nivel
    def _f(tr, te, f, y):
        c = cols or f
        X = np.log(tr[c].clip(lower=1e-5))
        e = est(); e.fit(X, np.log(tr[y].clip(lower=1e-5)))
        return np.exp(e.predict(np.log(te[c].clip(lower=1e-5))))
    return _f


HAR = ["har_1d", "har_5d", "har_22d"]
VOLF = HAR + ["vix_nivel_t1", "abs_ret_bova11_t1", "abs_ret_sp500_t1"]
ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10))
hgb = lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=50)
logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
hgbc = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=50)

def rel(a, b):
    return f"{(1 - a / b) * 100:+.1f}%"

pd.set_option("display.width", 200)
print("=" * 80)
print("1) AMPLITUDE (high-low)/open  | walk-forward mensal, teste desde 2014")
r = walk_forward(vol, VOLF, "amplitude", {
    "media": media,
    "HAR_ols_log": log_fit(LinearRegression, HAR),
    "HAR+VIX_ols_log": log_fit(LinearRegression, HAR + ["vix_nivel_t1"]),
    "ridge_todas": fit_pred(ridge),
    "hgb_todas": fit_pred(hgb),
}, min_train=750, start="2014-01")
for per, sub in [("2014-2026", r), ("ult. 2 anos", r[r.pregao >= "2024-10-17"])]:
    y = sub["amplitude"]
    b = (y - sub["media"]).abs().mean()
    print(f"  [{per}] n={len(sub)}  MAE media={b*100:.3f}pp")
    for k in ["HAR_ols_log", "HAR+VIX_ols_log", "ridge_todas", "hgb_todas"]:
        mae = (y - sub[k]).abs().mean()
        print(f"     {k:16s} MAE={mae*100:.3f}pp  ganho vs media={rel(mae,b)}  spearman={spearmanr(sub[k],y)[0]:.2f}")

print("=" * 80)
print("2) DIA LATERAL (|c-o|/open < 1%) | prob. calibrada")
vl = vol.copy(); vl["lateral"] = vl["lateral"]
r = walk_forward(vl, VOLF, "lateral", {
    "base": media,
    "logit_HAR+VIX": fit_proba(logit, HAR + ["vix_nivel_t1"]),
    "logit_todas": fit_proba(logit),
    "hgb_todas": fit_proba(hgbc),
}, min_train=750, start="2014-01")
for per, sub in [("2014-2026", r), ("ult. 2 anos", r[r.pregao >= "2024-10-17"])]:
    y = sub["lateral"].astype(int)
    bb = brier_score_loss(y, sub["base"])
    print(f"  [{per}] n={len(sub)} taxa real={y.mean():.1%}  Brier base={bb:.4f}")
    for k in ["logit_HAR+VIX", "logit_todas", "hgb_todas"]:
        br = brier_score_loss(y, sub[k])
        print(f"     {k:14s} Brier={br:.4f} ganho={rel(br,bb)} AUC={roc_auc_score(y, sub[k]):.3f}")
    # calibracao por quintil do melhor
    q = pd.qcut(sub["logit_HAR+VIX"], 5, labels=False, duplicates="drop")
    tab = sub.assign(q=q).groupby("q").agg(prob_media=("logit_HAR+VIX", "mean"), taxa_real=("lateral", "mean"), n=("lateral", "size"))
    print("     calibracao logit_HAR+VIX por quintil:\n" + tab.round(3).to_string().replace("\n", "\n       "))

print("=" * 80)
print("3) GAP de abertura | overnight ate 9h | teste apos 1 ano de treino")
GF = ["var_on_EWZ", "var_on_ES_FUT", "var_on_VIX", "var_on_BRENT_FUT"]
r = walk_forward(gap, GF, "gap", {
    "zero": zero,
    "ridge_EWZ": fit_pred(lambda: Ridge(alpha=1e-4), ["var_on_EWZ"]),
    "ridge_todas": fit_pred(lambda: Ridge(alpha=1e-4)),
    "ridge_sem_brent": fit_pred(lambda: Ridge(alpha=1e-4), GF[:3]),
}, min_train=230)
for per, sub in [("todos", r), ("sem 05/10/2026", r[r.pregao != "2026-10-05"])]:
    y = sub["gap"]; b = y.abs().mean()
    print(f"  [{per}] n={len(sub)}  MAE gap=0: {b*100:.3f}pp")
    for k in ["ridge_EWZ", "ridge_todas", "ridge_sem_brent"]:
        mae = (y - sub[k]).abs().mean()
        nz = y != 0
        dacc = (np.sign(sub[k][nz]) == np.sign(y[nz])).mean()
        r2 = 1 - ((y - sub[k]) ** 2).sum() / (y ** 2).sum()
        print(f"     {k:15s} MAE={mae*100:.3f}pp ganho={rel(mae,b)} R2oos={r2:.2f} direcao={dacc:.1%}")

print("=" * 80)
print("4) DIRECAO do dia (ret_intraday e ret_cc) usando o gap model / features overnight")
g2 = gap.copy()
for tgt in ["ret_intraday", "ret_cc"]:
    g2["sobe"] = (g2[tgt] > 0).astype(float)
    g2.loc[g2[tgt].isna(), "sobe"] = np.nan
    r = walk_forward(g2, GF, "sobe", {"base": media, "logit": fit_proba(logit)}, min_train=230)
    y = r["sobe"].astype(int)
    acc = ((r["logit"] > 0.5).astype(int) == y).mean()
    print(f"  {tgt:12s} n={len(r)} taxa de alta={y.mean():.1%}  acerto logit={acc:.1%}  AUC={roc_auc_score(y, r['logit']):.3f}")
