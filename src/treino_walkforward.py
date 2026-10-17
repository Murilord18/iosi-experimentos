"""
treino_walkforward.py
=====================
Fase D - Projeto IOSI. Treino e comparacao de modelos em walk-forward mensal.

Ponto de partida: src/previa_fase_d.py (previa feita fora deste projeto).

Validacao: para cada mes M de teste, treina com os pregoes ate o fim de M-1 e preve
os pregoes de M. Nunca embaralha. Janela de treino expansiva ou movel (5 anos).

Alvos e modelos:
    amplitude  (high-low)/open    baselines har_5d e har_22d (o proprio valor da feature);
                                  candidato principal HAR+VIX OLS em log; HAR OLS em log,
                                  Ridge, HistGradientBoosting e Random Forest com todas.
    lateral    |close-open|/open < 1%   baseline: taxa de dias laterais nos ultimos 252
                                  pregoes ate t-1; logisticas e HGB, cada um com janela
                                  expansiva e movel de 5 anos; Brier, AUC e calibracao.
    gap        open/close_ant - 1  baseline gap = 0; Ridge so com EWZ, com cada ativo a
                                  mais, com todos e com todos menos um (ablacao por ativo).
    ret_cc, ret_intraday          direcao com features overnight (referencia).

Metricas em tres recortes do periodo de teste: completo, ultimos 2 anos e sem 05/10/2026.

Saidas:
    relatorios/walkforward/metricas.csv, calibracao_lateral.csv, previsoes_<alvo>.csv,
    resumo.md
    models/<alvo>.joblib (vencedor reajustado com todos os dados) e models/vencedores.json

Uso:
    python src/treino_walkforward.py
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

TZ = ZoneInfo("America/Sao_Paulo")
HOJE = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
RAIZ = Path(__file__).parent.parent
DIR_DADOS = RAIZ / "data" / "datasets"
DIR_REL = RAIZ / "relatorios" / "walkforward"
DIR_MODELOS = RAIZ / "models"

HAR = ["har_1d", "har_5d", "har_22d"]
FEATS_VOL = HAR + ["vix_nivel_t1", "abs_ret_bova11_t1", "abs_ret_sp500_t1"]
ATIVOS_GAP = {"EWZ": "var_on_EWZ", "ES": "var_on_ES_FUT", "VIX": "var_on_VIX", "BRENT": "var_on_BRENT_FUT"}
FEATS_GAP = list(ATIVOS_GAP.values())

INICIO_TESTE_VOL = "2014-01"
MIN_TREINO_VOL = 750
MIN_TREINO_GAP = 230
JANELA_MOVEL_ANOS = 5
JANELA_TAXA_LATERAL = 252
DIA_ATIPICO = pd.Timestamp("2026-10-05")
EPS = 1e-5


# --------------------------------------------------------------------------- #
# Especificacao de modelos
# --------------------------------------------------------------------------- #
class Modelo:
    """Modelo do walk-forward: regra fixa (baseline) ou estimador do sklearn.

    tipo: "coluna" (preve o valor de uma coluna da propria linha), "zero", "media"
          (media do alvo no treino) ou "sklearn".
    """

    def __init__(self, tipo, cols=None, fabrica=None, log=False, proba=False, janela_anos=None):
        self.tipo, self.cols, self.fabrica = tipo, cols, fabrica
        self.log, self.proba, self.janela_anos = log, proba, janela_anos

    def _x(self, df):
        x = df[self.cols]
        return np.log(x.clip(lower=EPS)) if self.log else x

    def ajustar(self, tr: pd.DataFrame, alvo: str):
        if self.tipo == "media":
            return float(tr[alvo].mean())
        if self.tipo != "sklearn":
            return None
        est = self.fabrica()
        y = tr[alvo].astype(int) if self.proba else tr[alvo]
        est.fit(self._x(tr), np.log(y.clip(lower=EPS)) if self.log else y)
        return est

    def prever(self, ajuste, te: pd.DataFrame) -> np.ndarray:
        if self.tipo == "coluna":
            return te[self.cols[0]].to_numpy(dtype=float)
        if self.tipo == "zero":
            return np.zeros(len(te))
        if self.tipo == "media":
            return np.full(len(te), ajuste)
        if self.proba:
            return ajuste.predict_proba(self._x(te))[:, 1]
        p = ajuste.predict(self._x(te))
        return np.exp(p) if self.log else p

    def descricao(self) -> dict:
        return {"tipo": self.tipo, "features": self.cols, "transformacao_log": self.log,
                "saida": "probabilidade" if self.proba else "valor",
                "janela_treino": f"movel {self.janela_anos} anos" if self.janela_anos else "expansiva"}


def ridge(alpha):
    return lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))


def hgb_reg():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=200,
                                         min_samples_leaf=50, random_state=0)


def hgb_clf():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200,
                                          min_samples_leaf=50, random_state=0)


def rf_reg():
    return RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=50,
                                 max_features=0.5, n_jobs=-1, random_state=0)


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def preparar(df: pd.DataFrame, colunas: list, alvo: str) -> pd.DataFrame:
    df = df.dropna(subset=colunas + [alvo]).sort_values("pregao").reset_index(drop=True)
    df["mes"] = df["pregao"].dt.to_period("M")
    return df


def walk_forward(df: pd.DataFrame, colunas: list, alvo: str, modelos: dict,
                 min_treino: int, inicio: str = None) -> pd.DataFrame:
    """Previsoes fora da amostra de cada modelo, mes a mes."""
    df = preparar(df, colunas, alvo)
    prev = {k: np.full(len(df), np.nan) for k in modelos}
    for m in df["mes"].unique():
        if inicio and str(m) < inicio:
            continue
        te = (df["mes"] == m).to_numpy()
        for nome, mod in modelos.items():
            tr = df["mes"] < m
            if mod.janela_anos:
                tr &= df["pregao"] >= m.start_time - pd.DateOffset(years=mod.janela_anos)
            if (df["mes"] < m).sum() < min_treino:
                continue
            treino = df.loc[tr]
            prev[nome][te] = mod.prever(mod.ajustar(treino, alvo), df.loc[te])
    ok = ~np.isnan(np.column_stack(list(prev.values()))).any(axis=1)
    out = df.loc[ok, ["pregao", alvo]].reset_index(drop=True)
    for k, v in prev.items():
        out[k] = v[ok]
    return out


def recortes(prev: pd.DataFrame) -> dict:
    return {
        "completo": prev,
        "ultimos_2_anos": prev[prev["pregao"] > HOJE - pd.DateOffset(years=2)],
        "sem_05_10_2026": prev[prev["pregao"] != DIA_ATIPICO],
    }


def ganho(m: float, base: float) -> float:
    """Reducao percentual da perda em relacao ao baseline (positivo = melhor)."""
    return (1 - m / base) * 100


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #
def metricas_regressao(prev, alvo, modelos, bases) -> list:
    linhas = []
    for periodo, sub in recortes(prev).items():
        y = sub[alvo]
        mae_base = {b: (y - sub[b]).abs().mean() for b in bases}
        for nome in modelos:
            mae = (y - sub[nome]).abs().mean()
            linhas.append({"alvo": alvo, "modelo": nome, "periodo": periodo, "n": len(sub),
                           "mae_pp": mae * 100, "spearman": spearmanr(sub[nome], y).statistic
                           if sub[nome].std() > 0 else np.nan,
                           **{f"ganho_mae_vs_{b}_%": ganho(mae, mae_base[b]) for b in bases}})
    return linhas


def metricas_lateral(prev, modelos, base) -> list:
    linhas = []
    for periodo, sub in recortes(prev).items():
        y = sub["lateral"].astype(int)
        brier_base = brier_score_loss(y, sub[base])
        for nome in modelos:
            p = sub[nome]
            top = p >= p.quantile(0.8)
            linhas.append({"alvo": "lateral", "modelo": nome, "periodo": periodo, "n": len(sub),
                           "taxa_real": y.mean(), "brier": brier_score_loss(y, p),
                           f"ganho_brier_vs_{base}_%": ganho(brier_score_loss(y, p), brier_base),
                           "auc": roc_auc_score(y, p) if p.nunique() > 1 else np.nan,
                           "precisao_p>=0.5": y[p >= 0.5].mean() if (p >= 0.5).any() else np.nan,
                           "frac_p>=0.5": (p >= 0.5).mean(),
                           "taxa_real_top20%": y[top].mean()})
    return linhas


def calibracao(prev, modelos) -> pd.DataFrame:
    tabs = []
    for periodo, sub in recortes(prev).items():
        for nome in modelos:
            q = pd.qcut(sub[nome].rank(method="first"), 5, labels=False) + 1
            t = sub.assign(quintil=q).groupby("quintil").agg(
                prob_media=(nome, "mean"), taxa_real=("lateral", "mean"), n=("lateral", "size"))
            tabs.append(t.reset_index().assign(modelo=nome, periodo=periodo))
    return pd.concat(tabs, ignore_index=True)[["modelo", "periodo", "quintil", "prob_media", "taxa_real", "n"]]


def metricas_gap(prev, alvo, modelos) -> list:
    linhas = []
    for periodo, sub in recortes(prev).items():
        y = sub[alvo]
        mae_zero = y.abs().mean()
        nz = y != 0
        for nome in modelos:
            e = y - sub[nome]
            linhas.append({"alvo": alvo, "modelo": nome, "periodo": periodo, "n": len(sub),
                           "mae_pp": e.abs().mean() * 100,
                           "ganho_mae_vs_zero_%": ganho(e.abs().mean(), mae_zero),
                           "r2_oos": 1 - (e ** 2).sum() / (y ** 2).sum(),
                           "acerto_direcao": (np.sign(sub[nome][nz]) == np.sign(y[nz])).mean()
                           if nome != "zero" else np.nan})
    return linhas


# --------------------------------------------------------------------------- #
# Experimentos
# --------------------------------------------------------------------------- #
def experimento_amplitude(vol):
    modelos = {
        "har_5d (baseline)":  Modelo("coluna", ["har_5d"]),
        "har_22d (baseline)": Modelo("coluna", ["har_22d"]),
        "media_historica":    Modelo("media"),
        "HAR_ols_log":        Modelo("sklearn", HAR, LinearRegression, log=True),
        "HAR+VIX_ols_log":    Modelo("sklearn", HAR + ["vix_nivel_t1"], LinearRegression, log=True),
        "ridge_todas":        Modelo("sklearn", FEATS_VOL, ridge(10)),
        "hgb_todas":          Modelo("sklearn", FEATS_VOL, hgb_reg),
        "rf_todas":           Modelo("sklearn", FEATS_VOL, rf_reg),
    }
    prev = walk_forward(vol, FEATS_VOL, "amplitude", modelos, MIN_TREINO_VOL, INICIO_TESTE_VOL)
    met = metricas_regressao(prev, "amplitude", modelos, ["har_5d (baseline)", "har_22d (baseline)"])
    return modelos, prev, pd.DataFrame(met)


def experimento_lateral(vol):
    candidatos = {
        "logit_HAR+VIX": (HAR + ["vix_nivel_t1"], logit, True),
        "logit_todas":   (FEATS_VOL, logit, True),
        "hgb_todas":     (FEATS_VOL, hgb_clf, True),
        "taxa_media":    (None, None, False),
    }
    modelos = {"taxa_252d (baseline)": Modelo("coluna", ["taxa_lateral_252d"])}
    for nome, (cols, fab, sk) in candidatos.items():
        for rot, anos in (("expansiva", None), ("movel_5a", JANELA_MOVEL_ANOS)):
            modelos[f"{nome} | {rot}"] = (Modelo("sklearn", cols, fab, proba=True, janela_anos=anos) if sk
                                          else Modelo("media", janela_anos=anos))
    prev = walk_forward(vol, FEATS_VOL + ["taxa_lateral_252d"], "lateral", modelos,
                        MIN_TREINO_VOL, INICIO_TESTE_VOL)
    met = metricas_lateral(prev, modelos, "taxa_252d (baseline)")
    cal = calibracao(prev, list(modelos))
    return modelos, prev, pd.DataFrame(met), cal


def experimento_gap(gap):
    ewz = [ATIVOS_GAP["EWZ"]]
    modelos = {"zero (baseline)": Modelo("zero"),
               "ridge_EWZ": Modelo("sklearn", ewz, ridge(1e-4))}
    for a, c in ATIVOS_GAP.items():
        if a != "EWZ":
            modelos[f"ridge_EWZ+{a}"] = Modelo("sklearn", ewz + [c], ridge(1e-4))
    modelos["ridge_todas"] = Modelo("sklearn", FEATS_GAP, ridge(1e-4))
    for a, c in ATIVOS_GAP.items():
        modelos[f"ridge_todas_sem_{a}"] = Modelo("sklearn", [x for x in FEATS_GAP if x != c], ridge(1e-4))
    prev = walk_forward(gap, FEATS_GAP, "gap", modelos, MIN_TREINO_GAP)
    return modelos, prev, pd.DataFrame(metricas_gap(prev, "gap", modelos))


def experimento_retorno(gap):
    """Referencia: retorno do dia com as features overnight (Ridge contra retorno = 0)."""
    tabs, prevs = [], {}
    for alvo in ("ret_cc", "ret_intraday"):
        modelos = {"zero (baseline)": Modelo("zero"),
                   "ridge_todas": Modelo("sklearn", FEATS_GAP, ridge(1.0)),
                   "ridge_EWZ": Modelo("sklearn", [ATIVOS_GAP["EWZ"]], ridge(1.0))}
        prevs[alvo] = walk_forward(gap, FEATS_GAP, alvo, modelos, MIN_TREINO_GAP)
        tabs.append(pd.DataFrame(metricas_gap(prevs[alvo], alvo, modelos)))
    return prevs, pd.concat(tabs, ignore_index=True)


# --------------------------------------------------------------------------- #
# Vencedores
# --------------------------------------------------------------------------- #
def salvar_vencedor(alvo, modelos, met, criterio, menor_melhor, df, colunas):
    comp = met[met["periodo"] == "completo"].set_index("modelo")[criterio]
    nome = comp.idxmin() if menor_melhor else comp.idxmax()
    mod = modelos[nome]
    treino = preparar(df, colunas, alvo)
    if mod.janela_anos:
        treino = treino[treino["pregao"] >= treino["pregao"].max() - pd.DateOffset(years=mod.janela_anos)]
    ajuste = mod.ajustar(treino, alvo)
    vence_em = {p: bool((met[met["periodo"] == p].set_index("modelo")[criterio].idxmin() if menor_melhor
                         else met[met["periodo"] == p].set_index("modelo")[criterio].idxmax()) == nome)
                for p in met["periodo"].unique()}
    meta = {"alvo": alvo, "modelo": nome, **mod.descricao(), "criterio": f"{criterio} (periodo completo)",
            "tambem_vence_em": vence_em, "n_treino": len(treino),
            "treino_de": str(treino["pregao"].min().date()), "treino_ate": str(treino["pregao"].max().date()),
            "metricas": met[met["modelo"] == nome].drop(columns=["alvo", "modelo"]).round(6)
            .to_dict(orient="records")}
    joblib.dump({**meta, "estimador": ajuste}, DIR_MODELOS / f"{alvo}.joblib")
    return meta


# --------------------------------------------------------------------------- #
def fmt_tabela(met: pd.DataFrame, cols: list) -> str:
    t = met.pivot_table(index="modelo", columns="periodo", values=cols, sort=False)
    t = t.reindex(columns=["completo", "ultimos_2_anos", "sem_05_10_2026"], level=1)
    return t.round(4).to_string()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    DIR_REL.mkdir(parents=True, exist_ok=True)
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)

    vol = pd.read_csv(DIR_DADOS / "dataset_vol_lateral.csv", parse_dates=["pregao"])
    gap = pd.read_csv(DIR_DADOS / "dataset_gap_9h.csv", parse_dates=["pregao"])
    # baseline justo de lateral: taxa nos ultimos 252 pregoes com rotulo, ate t-1
    vol["taxa_lateral_252d"] = vol["lateral"].rolling(JANELA_TAXA_LATERAL, min_periods=200).mean().shift(1)

    logger.info("Amplitude...")
    mod_amp, prev_amp, met_amp = experimento_amplitude(vol)
    logger.info("Lateral...")
    mod_lat, prev_lat, met_lat, cal_lat = experimento_lateral(vol)
    logger.info("Gap...")
    mod_gap, prev_gap, met_gap = experimento_gap(gap)
    logger.info("Retorno (referencia)...")
    prev_ret, met_ret = experimento_retorno(gap)

    metricas = pd.concat([met_amp, met_lat, met_gap, met_ret], ignore_index=True)
    metricas.to_csv(DIR_REL / "metricas.csv", index=False, float_format="%.6f")
    cal_lat.to_csv(DIR_REL / "calibracao_lateral.csv", index=False, float_format="%.6f")
    for alvo, p in (("amplitude", prev_amp), ("lateral", prev_lat), ("gap", prev_gap), *prev_ret.items()):
        p.to_csv(DIR_REL / f"previsoes_{alvo}.csv", index=False, float_format="%.6f")

    vencedores = {
        "amplitude": salvar_vencedor("amplitude", mod_amp, met_amp, "mae_pp", True, vol, FEATS_VOL),
        "lateral": salvar_vencedor("lateral", mod_lat, met_lat, "brier", True, vol,
                                   FEATS_VOL + ["taxa_lateral_252d"]),
        "gap": salvar_vencedor("gap", mod_gap, met_gap, "mae_pp", True, gap, FEATS_GAP),
    }
    (DIR_MODELOS / "vencedores.json").write_text(json.dumps(vencedores, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")

    periodo_amp = f"{prev_amp['pregao'].min().date()} a {prev_amp['pregao'].max().date()}"
    periodo_gap = f"{prev_gap['pregao'].min().date()} a {prev_gap['pregao'].max().date()}"
    cal_principal = cal_lat[cal_lat["modelo"].isin(["logit_HAR+VIX | expansiva", "logit_HAR+VIX | movel_5a"])]
    secoes = [
        f"# Fase D - walk-forward mensal ({HOJE.date()})\n",
        f"## Amplitude (teste {periodo_amp})\n",
        fmt_tabela(met_amp, ["mae_pp", "ganho_mae_vs_har_22d (baseline)_%", "ganho_mae_vs_har_5d (baseline)_%", "spearman"]),
        f"\n## Lateral (teste {periodo_amp})\n",
        fmt_tabela(met_lat, ["brier", "ganho_brier_vs_taxa_252d (baseline)_%", "auc", "taxa_real_top20%"]),
        "\n### Calibracao por quintil (logit_HAR+VIX)\n",
        cal_principal.pivot_table(index=["modelo", "quintil"], columns="periodo", values=["prob_media", "taxa_real"],
                                  sort=False).round(3).to_string(),
        f"\n## Gap (teste {periodo_gap})\n",
        fmt_tabela(met_gap, ["mae_pp", "ganho_mae_vs_zero_%", "r2_oos", "acerto_direcao"]),
        "\n## Retorno do dia (referencia)\n",
        met_ret.pivot_table(index=["alvo", "modelo"], columns="periodo",
                            values=["mae_pp", "ganho_mae_vs_zero_%", "acerto_direcao"], sort=False)
        .round(4).to_string(),
        "\n## Vencedores (models/)\n",
        "\n".join(f"- {a}: {v['modelo']} ({v['janela_treino']}); vence tambem em: {v['tambem_vence_em']}; "
                  f"treino {v['treino_de']} a {v['treino_ate']} (n = {v['n_treino']})"
                  for a, v in vencedores.items()),
    ]
    texto = "\n".join(secoes)
    (DIR_REL / "resumo.md").write_text(texto + "\n", encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
