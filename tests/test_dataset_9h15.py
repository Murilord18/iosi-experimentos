"""
Teste point-in-time da Fase C.

Para cada pregao t amostrado, os dados brutos sao cortados no que era conhecido as 9h
de t (barras diarias com data < t e candles de 1h encerrados ate 9h), o dataset e
remontado e as features da linha t precisam ser identicas as do dataset completo.
Se alguma feature usasse dado posterior as 9h de t, o valor mudaria com o corte.

Baixa os dados do Yahoo uma vez por execucao (~1 min).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import dataset_9h15 as ds  # noqa: E402

N_AMOSTRA = 40
DATAS_FIXAS = ["2025-10-24", "2025-10-27", "2026-10-05", "2026-10-15", "2026-10-16"]


@pytest.fixture(scope="module")
def dados():
    return ds.baixar_dados()


def amostra(indice: pd.DatetimeIndex, minimo: pd.Timestamp) -> list:
    candidatos = indice[indice >= minimo]
    rng = np.random.default_rng(42)
    sorteio = list(candidatos[np.sort(rng.choice(len(candidatos), N_AMOSTRA, replace=False))])
    fixas = [pd.Timestamp(d) for d in DATAS_FIXAS if pd.Timestamp(d) in candidatos]
    return sorted(set(sorteio + fixas))


def test_vol_point_in_time(dados):
    datas = amostra(dados["sessoes"], dados["sessoes"][30])
    assert ds.verificar_point_in_time(ds.montar_vol, dados, datas, ds.FEATS_VOL) == []


def test_gap_point_in_time(dados):
    datas = amostra(ds.montar_gap(dados).index, dados["inicio_1h"])
    assert ds.verificar_point_in_time(ds.montar_gap, dados, datas, ds.FEATS_GAP) == []


def test_verificacao_detecta_vazamento(dados):
    """Controle: uma feature com a amplitude do proprio dia t precisa ser apanhada."""
    def montar_com_vazamento(d):
        df = ds.montar_vol(d)
        df["har_1d"] = df["amplitude"]
        return df
    datas = amostra(dados["sessoes"], dados["sessoes"][30])
    assert len(ds.verificar_point_in_time(montar_com_vazamento, dados, datas, ["har_1d"])) > 0


def test_linha_de_hoje_tem_features(dados):
    """Na producao (9h15), a linha do dia precisa existir com as features preenchidas."""
    ultimo = dados["sessoes"][-1]
    vol = ds.montar_vol(ds.truncar(dados, ultimo))
    gap = ds.montar_gap(ds.truncar(dados, ultimo))
    assert vol.loc[ultimo, ds.FEATS_VOL].notna().all()
    assert gap.loc[ultimo, ds.FEATS_GAP].notna().all()
    assert vol.loc[ultimo, ["gap", "amplitude"]].isna().all()   # alvo ainda desconhecido


def test_barra_reconstruida_marcada(dados):
    """Pregao sem barra diaria (removida aqui de proposito) e montado com os candles de 1h.

    Nao depende de uma data real sem barra: o Yahoo pode preencher a lacuna depois
    (foi o que aconteceu com 15/10/2026).
    """
    t = dados["sessoes"][-2]
    sem_barra = {**dados, "diario": {**dados["diario"], "BOVA11": dados["diario"]["BOVA11"].drop(t)}}
    vol = ds.montar_vol(sem_barra)
    assert vol.loc[t, "fonte_barra"] == "agregado_1h"
    assert vol.loc[t, ["gap", "amplitude", "lateral"]].notna().all()
    real = ds.montar_vol(dados).loc[t, "amplitude"]
    assert abs(vol.loc[t, "amplitude"] / real - 1) < 0.02     # High/Low dos candles de 1h ~ barra diaria
