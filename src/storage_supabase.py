"""
storage_supabase.py
===================
Projeto IOSI - copia dos dados do robo no Supabase (tabelas em supabase/schema.sql).

Os CSVs e JSONs em data/ continuam sendo a fonte principal: este modulo NUNCA levanta
excecao. Sem SUPABASE_URL ou SUPABASE_SERVICE_ROLE_KEY no ambiente, ou se o Supabase
falhar, so registra um aviso no log e devolve False.

A chave pode ser do formato novo (sb_secret_...) ou a legacy (JWT service_role): o
cliente manda a mesma chave nos cabecalhos apikey e Authorization, aceitos nos dois casos.

Tabelas:
    snapshots  -> upsert por (ativo, tipo, inicio)      upsert_snapshots()
    previsoes  -> upsert por data; previsao das 9h15    upsert_previsao()
                  e conferencia das 19h na mesma linha   upsert_historico()
    execucoes  -> uma linha por script rodado            Execucao (with)

Rodar duas vezes nao duplica nada: o upsert atualiza a linha que ja existe.
"""

import logging
import math
import os
import traceback
from datetime import date, datetime

import numpy as np
import pandas as pd

logger = logging.getLogger("storage_supabase")

LOTE = 500                  # linhas por requisicao
TIMEOUT_S = 30

COLUNAS_SNAPSHOT = ["ativo", "ticker", "tipo", "inicio", "fim", "open", "high", "low", "close",
                    "volume", "fonte", "coletado_em"]
CHAVE_SNAPSHOT = "ativo,tipo,inicio"

# mesmas colunas (e ordem) do historico_previsoes.csv (validador_19h.COLUNAS)
COLUNAS_HISTORICO = [
    "data", "status", "validado_em", "simulacao", "previsao_gerada_em", "fonte_gap_card", "fonte_candle",
    "fech_anterior", "gap_prev_9h", "gap_prev_9h15", "gap_prev", "abertura_prev", "amplitude_prev", "faixa_inf", "faixa_sup", "prob_lateral",
    "open", "high", "low", "close", "gap_real", "amplitude_real", "abs_intraday_real", "lateral_real",
    "erro_gap_9h", "erro_gap_9h15", "erro_gap", "erro_abertura_rs", "erro_amplitude", "brier_lateral", "max_dentro_faixa", "min_dentro_faixa",
    "amplitude_har_5d", "erro_amplitude_har_5d", "erro_gap_zero", "taxa_lateral_252d", "brier_lateral_taxa_252d",
]
# campos do JSON da previsao que ja existem no historico com outro nome
JSON_PARA_COLUNA = {
    "status": "status_previsao", "gerado_em": "previsao_gerada_em", "gap_previsto": "gap_prev",
    "abertura_esperada": "abertura_prev", "amplitude_prevista": "amplitude_prev",
    "faixa_inferior": "faixa_inf", "faixa_superior": "faixa_sup",
}
COLUNAS_SO_JSON = [
    "status_previsao", "data_fech_anterior", "fonte_fech_anterior", "amplitude_prevista_rs",
    "faixa_k_inf", "faixa_k_sup", "faixa_cobertura_alvo", "faixa_cobertura_card",
    "baselines", "modelos", "features", "overnight_9h", "overnight_9h15",
    "alerta_formatado_telegram", "arquivo_previsao",
]
COLUNAS_PREVISOES = set(COLUNAS_HISTORICO) | set(COLUNAS_SO_JSON)
COLUNAS_TIMESTAMP = {"validado_em", "previsao_gerada_em", "coletado_em"}
COLUNAS_INTEIRAS = {"volume", "lateral_real", "max_dentro_faixa", "min_dentro_faixa"}

_cliente = None
_aviso_config_dado = False


def _env(nome: str) -> str:
    return (os.environ.get(nome) or "").strip()


def configurado() -> bool:
    return bool(_env("SUPABASE_URL") and _env("SUPABASE_SERVICE_ROLE_KEY"))


def cliente():
    """Cliente do Supabase, ou None (sem configuracao ou falha ao criar)."""
    global _cliente, _aviso_config_dado
    if _cliente is not None:
        return _cliente
    if not configurado():
        if not _aviso_config_dado:
            logger.warning("Supabase desativado: SUPABASE_URL ou SUPABASE_SERVICE_ROLE_KEY ausente "
                           "(so os CSVs serao gravados)")
            _aviso_config_dado = True
        return None
    try:
        from supabase import ClientOptions, create_client
        _cliente = create_client(
            _env("SUPABASE_URL"), _env("SUPABASE_SERVICE_ROLE_KEY"),
            options=ClientOptions(postgrest_client_timeout=TIMEOUT_S,
                                  auto_refresh_token=False, persist_session=False))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supabase: falha ao criar o cliente: %s", exc)
        return None
    return _cliente


# ---------------------------------------------------------------------------
# Conversao para JSON (NaN -> null, numpy -> python, horarios -> ISO 8601)
# ---------------------------------------------------------------------------
def limpo(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, dict):
        return {str(k): limpo(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [limpo(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.isoformat()
    return v


def _timestamp(v):
    if v is None:
        return None
    try:
        return pd.Timestamp(v).isoformat()
    except (ValueError, TypeError):
        return v


def _booleano(v):
    if v is None or isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return True if s in ("true", "1", "1.0") else False if s in ("false", "0", "0.0") else None


def _linha(registro: dict, colunas, sem_nulos: bool) -> dict:
    """Converte um registro para a tabela; com sem_nulos, campos vazios nao sao enviados
    (o upsert nao apaga o que outra etapa ja gravou na mesma linha)."""
    saida = {}
    for k, v in registro.items():
        if k not in colunas:
            continue
        v = limpo(v)
        if v is not None:
            if k in COLUNAS_TIMESTAMP:
                v = _timestamp(v)
            elif k in COLUNAS_INTEIRAS:
                v = int(v)
            elif k == "simulacao":
                v = _booleano(v)
        if v is None and sem_nulos:
            continue
        saida[k] = v
    return saida


def _upsert(tabela: str, linhas: list, chave: str) -> bool:
    """Upsert em lotes. Linhas com conjuntos de campos diferentes vao em lotes separados
    (o PostgREST preencheria os campos faltantes com null)."""
    if not linhas:
        return True
    sb = cliente()
    if sb is None:
        return False
    try:
        from postgrest import ReturnMethod
        campos = chave.split(",")
        unicas = {tuple(l[c] for c in campos): l for l in linhas}      # mesma chave: fica a ultima
        grupos = {}
        for l in unicas.values():
            grupos.setdefault(tuple(sorted(l)), []).append(l)
        for grupo in grupos.values():
            for i in range(0, len(grupo), LOTE):
                (sb.table(tabela).upsert(grupo[i:i + LOTE], on_conflict=chave,
                                         returning=ReturnMethod.minimal).execute())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supabase: falha no upsert em %s (%d linhas): %s", tabela, len(linhas), exc)
        return False
    logger.info("Supabase: %d linhas em %s", len(unicas), tabela)
    return True


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------
def upsert_snapshots(dados: pd.DataFrame, arquivo: str = None) -> bool:
    """Linhas de um CSV de snapshot (colunas do coletor_snapshots.COLUNAS)."""
    try:
        linhas = []
        for r in dados.to_dict("records"):
            l = _linha(r, COLUNAS_SNAPSHOT, sem_nulos=False)
            l["arquivo"] = r.get("arquivo", arquivo)
            linhas.append(l)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supabase: snapshots nao convertidos: %s", exc)
        return False
    return _upsert("snapshots", linhas, CHAVE_SNAPSHOT)


# ---------------------------------------------------------------------------
# previsoes
# ---------------------------------------------------------------------------
def linha_previsao(p: dict, arquivo: str = None) -> dict:
    """JSON do predictor_9h15 -> linha de previsoes (sem os campos da conferencia)."""
    r = {JSON_PARA_COLUNA.get(k, k): v for k, v in p.items()}
    for k in ("amplitude_har_5d", "taxa_lateral_252d"):
        if k in (p.get("baselines") or {}):
            r[k] = p["baselines"][k]
    r["arquivo_previsao"] = arquivo
    return _linha(r, COLUNAS_PREVISOES, sem_nulos=True)


def upsert_previsao(p: dict, arquivo: str = None) -> bool:
    try:
        linha = linha_previsao(p, arquivo)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supabase: previsao nao convertida: %s", exc)
        return False
    return _upsert("previsoes", [linha], "data")


def upsert_historico(historico: pd.DataFrame) -> bool:
    """Linhas do historico_previsoes.csv (conferencia das 19h)."""
    try:
        linhas = [_linha(r, COLUNAS_HISTORICO, sem_nulos=True)
                  for r in historico.astype(object).to_dict("records")]
        for l in linhas:
            l["data"] = str(l["data"])[:10]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supabase: historico nao convertido: %s", exc)
        return False
    return _upsert("previsoes", linhas, "data")


# ---------------------------------------------------------------------------
# execucoes
# ---------------------------------------------------------------------------
class Execucao:
    """Registra um script do robo em execucoes:

        with Execucao("noite", "coleta") as ex:
            ...
            ex.falha("Falhas: BOVA11.SA")   # opcional; sem isso termina como sucesso

    Uma excecao dentro do bloco grava status falha com o traceback (e segue para cima).
    """

    def __init__(self, tipo: str, etapa: str, ativo: bool = True):
        self.tipo, self.etapa, self.ativo = tipo, etapa, ativo
        self.id = None
        self.mensagem = None

    def falha(self, mensagem: str) -> None:
        self.mensagem = mensagem

    def __enter__(self):
        if not self.ativo or (sb := cliente()) is None:
            return self
        linha = {"tipo": self.tipo, "etapa": self.etapa, "status": "em_andamento",
                 "inicio": pd.Timestamp.now(tz="UTC").isoformat(),
                 "origem": "github" if os.environ.get("GITHUB_ACTIONS") == "true" else "local",
                 "github_run_id": os.environ.get("GITHUB_RUN_ID")}
        try:
            self.id = sb.table("execucoes").insert(linha).execute().data[0]["id"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supabase: falha ao registrar inicio da execucao: %s", exc)
        return self

    def __exit__(self, tipo_exc, exc, tb):
        if self.id is None:
            return False
        if exc is not None and not (isinstance(exc, SystemExit) and exc.code in (0, None)):
            self.mensagem = "".join(traceback.format_exception(tipo_exc, exc, tb))[-4000:]
        fim = {"status": "falha" if self.mensagem else "sucesso", "mensagem_erro": self.mensagem,
               "fim": pd.Timestamp.now(tz="UTC").isoformat()}
        try:
            cliente().table("execucoes").update(fim).eq("id", self.id).execute()
        except Exception as e:  # noqa: BLE001
            logger.warning("Supabase: falha ao registrar fim da execucao %s: %s", self.id, e)
        return False
