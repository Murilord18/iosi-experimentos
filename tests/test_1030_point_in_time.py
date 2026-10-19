"""
Teste point-in-time da previsao das 10h30: nenhuma feature pode usar candle com fim > 10h30.

- sintetico (sem rede): as features das 10h30 (leitura_1030.features_1030) nao mudam quando
  os candles que terminam depois de 10h30 sao removidos ou adulterados; o candle que termina
  exatamente as 10h30 entra, o das 10h30-10h35 nao;
- controle: uma leitura com referencia 10h35 (vazamento proposital) e apanhada;
- dados reais (Yahoo, ~60 dias de 5m): o mesmo, dia a dia, para BOVA11, EWZ, ES=F e ^VIX;
- predictor_1030 de ponta a ponta: um snapshot com candles depois de 10h30 (como se o
  coletor tivesse falhado no corte) da a mesma previsao que o snapshot cortado.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import dataset_9h15 as ds  # noqa: E402
import leitura_1030 as l1030  # noqa: E402

T = pd.Timestamp("2026-10-16")
ATIVOS = ["BOVA11", *l1030.EXTERNOS]


def sinteticos(t: pd.Timestamp, seed: int = 0) -> dict:
    """Candles de 5m: BOVA11 das 10h05 as 16h55 e externos desde as 16h de t-1."""
    rng = np.random.default_rng(seed)
    out = {}
    for nome in ATIVOS:
        ini0 = l1030.referencia(t, pd.Timedelta(hours=10, minutes=5)) if nome == "BOVA11" \
            else l1030.referencia(t - pd.Timedelta(days=1), pd.Timedelta(hours=16))
        inicio = pd.date_range(ini0, l1030.referencia(t, pd.Timedelta(hours=16, minutes=50)), freq="5min")
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(inicio))))
        abre = np.r_[100, close[:-1]]
        out[nome] = pd.DataFrame({"inicio": inicio, "fim": inicio + l1030.DURACAO_5M, "open": abre,
                                  "high": np.maximum(abre, close) * 1.001, "low": np.minimum(abre, close) * 0.999,
                                  "close": close, "volume": rng.integers(1000, 5000, len(inicio)).astype(float)})
    return out


FECH = {nome: 99.0 for nome in ATIVOS}


def iguais(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.isclose(a[k], b[k], rtol=1e-12, atol=0, equal_nan=True) for k in a)


def cortar(candles: dict, ref: pd.Timestamp) -> dict:
    return {k: v[v["fim"] <= ref] for k, v in candles.items()}


def adulterar(candles: dict, ref: pd.Timestamp) -> dict:
    """Precos absurdos em todo candle que termina depois da referencia."""
    out = {}
    for k, v in candles.items():
        v = v.copy()
        depois = v["fim"] > ref
        v.loc[depois, ["open", "high", "low", "close"]] *= 1.5
        v.loc[depois, "volume"] *= 10
        out[k] = v
    return out


def test_sintetico_ignora_candles_depois_das_10h30():
    c = sinteticos(T)
    ref = l1030.referencia(T)
    completo = l1030.features_1030(c, FECH, T)
    assert completo["preco_1030"] == c["BOVA11"].loc[c["BOVA11"]["fim"] == ref, "close"].iloc[0]
    assert iguais(completo, l1030.features_1030(cortar(c, ref), FECH, T))
    assert iguais(completo, l1030.features_1030(adulterar(c, ref), FECH, T))
    lb = l1030.leitura_bova(c["BOVA11"], T)
    assert lb["preco_1030_fim"] == ref and lb["n_candles_ate_1030"] == 5     # 10h05 ... 10h25


def test_controle_detecta_vazamento():
    """Uma leitura com referencia 10h35 usa um candle com fim > 10h30 e tem de ser apanhada."""
    c = sinteticos(T, seed=1)
    ref = l1030.referencia(T)

    def com_vazamento(candles):
        lb = l1030.leitura_bova(candles["BOVA11"], T, ref + l1030.DURACAO_5M)
        return {k: v for k, v in lb.items() if k != "preco_1030_fim"}

    assert not iguais(com_vazamento(c), com_vazamento(cortar(c, ref)))
    assert not iguais(com_vazamento(c), com_vazamento(adulterar(c, ref)))


@pytest.fixture(scope="module")
def candles_reais():
    c = {"BOVA11": l1030.de_yahoo(ds.baixar("BOVA11.SA", period="60d", interval="5m"))}
    for nome, tk in l1030.EXTERNOS.items():
        c[nome] = l1030.de_yahoo(ds.baixar(tk, period="60d", interval="5m", prepost=True))
    return c


def test_dados_reais_point_in_time(candles_reais):
    dias = candles_reais["BOVA11"]["inicio"].dt.tz_localize(None).dt.normalize().unique()
    assert len(dias) >= 30
    com_leitura = 0
    for t in dias:
        t = pd.Timestamp(t)
        ref = l1030.referencia(t)
        completo = l1030.features_1030(candles_reais, FECH, t)
        lb = l1030.leitura_bova(candles_reais["BOVA11"], t)
        if not lb:
            # ex.: 31/07/2026, o Yahoo so tem candles do BOVA11 a partir das 12h55
            assert "preco_1030" not in completo, t.date()
            continue
        com_leitura += 1
        assert iguais(completo, l1030.features_1030(cortar(candles_reais, ref), FECH, t)), t.date()
        assert iguais(completo, l1030.features_1030(adulterar(candles_reais, ref), FECH, t)), t.date()
        assert lb["preco_1030_fim"] <= ref
    assert com_leitura >= 0.9 * len(dias)


def gravar_snapshot(pasta: Path, t: pd.Timestamp, candles: dict, fech: dict) -> None:
    linhas = []
    for nome, c in candles.items():
        for r in c.itertuples():
            linhas.append({"ativo": nome, "ticker": nome, "tipo": "5m",
                           "inicio": r.inicio.strftime("%Y-%m-%d %H:%M:%S%z"),
                           "fim": r.fim.strftime("%Y-%m-%d %H:%M:%S%z"), "open": r.open, "high": r.high,
                           "low": r.low, "close": r.close, "volume": r.volume, "fonte": "teste", "coletado_em": ""})
        linhas.append({"ativo": nome, "ticker": nome, "tipo": "1d", "inicio": fech[nome][0], "fim": fech[nome][0],
                       "open": 1, "high": 1, "low": 1, "close": fech[nome][1], "volume": 1, "fonte": "teste",
                       "coletado_em": ""})
    pasta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(pasta / f"{t:%Y-%m-%d}_10h30.csv", index=False)


def test_predictor_1030_ponta_a_ponta(candles_reais, tmp_path):
    import predictor_1030 as p1030
    dados = ds.baixar_dados()
    t = pd.Timestamp(candles_reais["BOVA11"]["inicio"].dt.tz_localize(None).dt.normalize().iloc[-1])
    ref = l1030.referencia(t)
    dia = {k: v[v["inicio"] >= l1030.referencia(t - pd.Timedelta(days=4), pd.Timedelta(hours=16))]
           for k, v in candles_reais.items()}
    fech = {k: ("2000-01-01", 1.0) for k in dia}
    gravar_snapshot(tmp_path / "vazado", t, adulterar(dia, ref), fech)
    gravar_snapshot(tmp_path / "cortado", t, cortar(dia, ref), fech)

    a = p1030.prever(t, tmp_path / "vazado", dados)
    b = p1030.prever(t, tmp_path / "cortado", dados)
    assert a["status"] == b["status"] == "ok"
    numericos = [k for k, v in a.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    assert numericos and iguais({k: a[k] for k in numericos}, {k: b[k] for k in numericos})
    assert iguais(a["features_1030"], b["features_1030"])
    assert pd.Timestamp(a["preco_1030_fim"]) <= ref
