"""
Testes do storage_supabase.py, sem rede: um cliente falso registra as chamadas.

- sem SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY: nada levanta excecao, tudo devolve False;
- falha do Supabase: so aviso, devolve False;
- conversao: NaN -> null, numpy -> python, campos do JSON renomeados para as colunas do historico;
- upsert: chave repetida no mesmo envio fica uma vez; campos vazios nao sao enviados;
- Execucao: grava em_andamento no inicio e sucesso/falha no fim.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import storage_supabase as sb  # noqa: E402


class Consulta:
    def __init__(self, cliente, tabela):
        self.cliente, self.tabela = cliente, tabela

    def _registra(self, op, dados, **kw):
        if self.cliente.falhar:
            raise RuntimeError("supabase fora do ar")
        self.cliente.chamadas.append((self.tabela, op, dados, kw))
        return self

    def upsert(self, dados, **kw):
        return self._registra("upsert", dados, **kw)

    def insert(self, dados, **kw):
        return self._registra("insert", dados, **kw)

    def update(self, dados, **kw):
        return self._registra("update", dados, **kw)

    def eq(self, coluna, valor):
        self.cliente.chamadas[-1][3]["eq"] = (coluna, valor)
        return self

    def execute(self):
        return type("Resp", (), {"data": [{"id": 7}]})()


class ClienteFalso:
    def __init__(self, falhar=False):
        self.chamadas, self.falhar = [], falhar

    def table(self, nome):
        return Consulta(self, nome)


@pytest.fixture
def falso(monkeypatch):
    c = ClienteFalso()
    monkeypatch.setattr(sb, "_cliente", c)
    return c


SNAP = pd.DataFrame({
    "ativo": ["BOVA11", "BOVA11", "BOVA11"], "ticker": "BOVA11.SA", "tipo": ["1d", "5m", "5m"],
    "inicio": ["2026-10-16", "2026-10-16 10:05:00-0300", "2026-10-16 10:05:00-0300"],
    "fim": ["2026-10-16", "2026-10-16 10:10:00-0300", "2026-10-16 10:10:00-0300"],
    "open": [200.22, 200.22, 200.3], "high": [203.19, 201.86, 201.9], "low": [200.17, 200.17, 200.2],
    "close": [202.3, 201.59, 201.6], "volume": [4052343.0, 20962.0, 21000.0], "fonte": "yahoo",
    "coletado_em": "2026-10-16 19:00:01-0300",
})


def test_sem_configuracao_nao_quebra(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setattr(sb, "_cliente", None)
    assert sb.cliente() is None
    assert sb.upsert_snapshots(SNAP, "x.csv") is False
    assert sb.upsert_previsao({"data": "2026-10-19", "status": "ok"}) is False
    with sb.Execucao("noite", "coleta") as ex:
        ex.falha("qualquer")
    assert ex.id is None


def test_falha_do_supabase_vira_aviso(monkeypatch, caplog):
    monkeypatch.setattr(sb, "_cliente", ClienteFalso(falhar=True))
    assert sb.upsert_snapshots(SNAP, "x.csv") is False
    with sb.Execucao("manha", "previsao") as ex:
        pass
    assert ex.id is None
    assert "supabase fora do ar" in caplog.text


def test_limpo():
    assert sb.limpo(np.float64("nan")) is None
    assert sb.limpo(float("inf")) is None
    assert sb.limpo(np.int64(3)) == 3 and type(sb.limpo(np.int64(3))) is int
    assert sb.limpo({"a": np.nan, "b": [np.float32(1.5)]}) == {"a": None, "b": [1.5]}


def test_snapshots_sem_duplicar(falso):
    assert sb.upsert_snapshots(SNAP, "2026-10-16_19h.csv")
    (tabela, op, linhas, kw), = falso.chamadas
    assert (tabela, op, kw["on_conflict"]) == ("snapshots", "upsert", "ativo,tipo,inicio")
    assert len(linhas) == 2                                     # candle de 5m repetido: fica o ultimo
    assert linhas[1]["close"] == 201.6
    assert linhas[0]["volume"] == 4052343 and type(linhas[0]["volume"]) is int
    assert linhas[0]["coletado_em"] == "2026-10-16T19:00:01-03:00"
    assert linhas[0]["arquivo"] == "2026-10-16_19h.csv"


def test_previsao_renomeia_campos(falso):
    p = {"data": "2026-10-19", "status": "ok", "gerado_em": "2026-10-19 09:17:30-0300",
         "gap_previsto": 0.003, "abertura_esperada": 202.0, "gap_prev_9h15": None,
         "baselines": {"amplitude_har_5d": 0.0154, "gap_zero": 0.0, "taxa_lateral_252d": 0.75},
         "features": {"har_5d": np.float64(0.0154)}, "simulacao": False, "campo_novo": 1}
    assert sb.upsert_previsao(p, "2026-10-19.json")
    (_, _, (linha,), kw), = falso.chamadas
    assert kw["on_conflict"] == "data"
    assert linha["status_previsao"] == "ok" and "status" not in linha
    assert linha["gap_prev"] == 0.003 and linha["abertura_prev"] == 202.0
    assert linha["previsao_gerada_em"] == "2026-10-19T09:17:30-03:00"
    assert linha["amplitude_har_5d"] == 0.0154 and linha["taxa_lateral_252d"] == 0.75
    assert "gap_prev_9h15" not in linha and "campo_novo" not in linha   # vazio e desconhecido: fora


def test_historico_nao_apaga_previsao(falso):
    h = pd.DataFrame([{"data": "2026-10-16", "status": "sem_previsao",
                       "validado_em": "2026-10-16 23:41:15-0300", "simulacao": np.nan, "gap_prev": np.nan},
                      {"data": "2026-10-19", "status": "ok", "validado_em": "2026-10-19 19:01:00-0300",
                       "simulacao": "False", "gap_prev": 0.003, "lateral_real": 1.0}])
    assert sb.upsert_historico(h)
    linhas = [l for (_, _, lote, _) in falso.chamadas for l in lote]
    assert len(falso.chamadas) == 2                             # campos diferentes: lotes separados
    assert linhas[0] == {"data": "2026-10-16", "status": "sem_previsao",
                         "validado_em": "2026-10-16T23:41:15-03:00"}
    assert linhas[1]["simulacao"] is False and linhas[1]["lateral_real"] == 1


def test_execucao(falso, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    with sb.Execucao("noite", "validacao") as ex:
        ex.falha("status sem_candle_real")
    inicio, fim = falso.chamadas
    assert inicio[:2] == ("execucoes", "insert")
    assert inicio[2]["origem"] == "github" and inicio[2]["status"] == "em_andamento"
    assert fim[1] == "update" and fim[3]["eq"] == ("id", 7)
    assert fim[2]["status"] == "falha" and fim[2]["mensagem_erro"] == "status sem_candle_real"

    falso.chamadas.clear()
    with pytest.raises(ValueError):
        with sb.Execucao("manha", "previsao"):
            raise ValueError("quebrou")
    assert falso.chamadas[1][2]["status"] == "falha"
    assert "ValueError: quebrou" in falso.chamadas[1][2]["mensagem_erro"]


class ClienteSchemaAntigo(ClienteFalso):
    """Supabase com o schema anterior: recusa colunas das 10h30 e o tipo manha_1030."""

    def table(self, nome):
        cliente = self

        class C(Consulta):
            def _registra(self, op, dados, **kw):
                linhas = dados if isinstance(dados, list) else [dados]
                if any(sb.COLUNAS_1030 & set(l) for l in linhas) or any(l.get("tipo") == "manha_1030" for l in linhas):
                    raise RuntimeError("PGRST204 coluna ou valor desconhecido")
                return super()._registra(op, dados, **kw)
        return C(cliente, nome)


def test_previsao_1030(falso):
    p = {"data": "2026-10-19", "status": "ok", "previsao_hora": "10h30", "gerado_em": "2026-10-19 10:31:10-0300",
         "abertura_real": 200.22, "preco_1030": 202.64, "preco_1030_fim": "2026-10-19 10:30:00-0300",
         "faixa_ajustada": False, "faixa_inferior": 196.8, "faixa_superior": 203.5,
         "features_1030": {"amp_ate_1030": 0.013}}
    assert sb.upsert_previsao(p, "2026-10-19_1030.json")
    (_, _, (linha,), _), = falso.chamadas
    assert linha["abertura_real_1030"] == 200.22 and "abertura_real" not in linha
    assert linha["preco_1030_fim"] == "2026-10-19T10:30:00-03:00"
    assert linha["faixa_ajustada"] is False and linha["faixa_inf"] == 196.8
    assert linha["features_1030"] == {"amp_ate_1030": 0.013} and linha["previsao_hora"] == "10h30"


def test_schema_antigo_reenvia_sem_colunas_1030(monkeypatch, caplog):
    c = ClienteSchemaAntigo()
    monkeypatch.setattr(sb, "_cliente", c)
    assert sb.upsert_previsao({"data": "2026-10-19", "status": "ok", "preco_1030": 202.64, "faixa_inferior": 196.8})
    (_, _, (linha,), _), = c.chamadas
    assert linha == {"data": "2026-10-19", "status_previsao": "ok", "faixa_inf": 196.8}
    assert "schema.sql" in caplog.text

    c.chamadas.clear()
    with sb.Execucao("manha_1030", "coleta") as ex:
        pass
    assert ex.id == 7 and c.chamadas[0][2]["tipo"] == "manha"
