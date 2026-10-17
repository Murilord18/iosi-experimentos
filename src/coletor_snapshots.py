"""
coletor_snapshots.py
====================
Projeto IOSI - coletor diario de snapshots ao vivo (insumo da Fase C).

O historico do Yahoo em 5m so cobre ~60 dias, e o de 1h so chega ate as 9h (o candle
das 9h as 10h fecha depois da abertura da B3). Para ter o preco das 9h15 no longo
prazo e preciso grava-lo todo dia.

Modos:
    manha (9h15 BRT): candles de 5m com pre-mercado de EWZ, ES=F, ^VIX e BZ=F, desde
                      as 16h BRT do dia util anterior (cobre o fechamento da B3 em D-1),
                      mais o ultimo fechamento diario (data < hoje) do BOVA11 e desses ativos.
    noite (19h BRT):  candle diario do BOVA11 do dia (open, high, low, close, volume) e os
                      candles de 5m do BOVA11 no pregao do dia.
                      Se o Yahoo nao trouxer a barra diaria de hoje, o candle e montado
                      com os candles de 5m do pregao (coluna "fonte" indica a origem).

Saida: data/snapshots/AAAA-MM-DD_9h15.csv e data/snapshots/AAAA-MM-DD_19h.csv, formato
longo, horarios em BRT. Nenhum arquivo e sobrescrito: se o do dia ja existir, o novo
ganha o sufixo _HHMMSS. Log acumulado em data/snapshots/coletor.log.

Uso:
    python src/coletor_snapshots.py manha
    python src/coletor_snapshots.py noite
    python src/coletor_snapshots.py manha --saida <pasta>   # teste, fora de data/snapshots
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

TZ = ZoneInfo("America/Sao_Paulo")
ALVO = ("BOVA11", "BOVA11.SA")
ATIVOS_5M = {
    "EWZ":       "EWZ",
    "ES_FUT":    "ES=F",
    "VIX":       "^VIX",
    "BRENT_FUT": "BZ=F",
}
INICIO_JANELA = pd.Timedelta(hours=16)      # 16h BRT do dia util anterior
DURACAO_5M = pd.Timedelta(minutes=5)
TENTATIVAS = 3
ESPERA_S = 15
COLUNAS = ["ativo", "ticker", "tipo", "inicio", "fim", "open", "high", "low", "close",
           "volume", "fonte", "coletado_em"]

SAIDA_PADRAO = Path(__file__).parent.parent / "data" / "snapshots"
logger = logging.getLogger("coletor")


def configurar_log(pasta: Path) -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(pasta / "coletor.log", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)


def baixar(ticker: str, **kwargs) -> pd.DataFrame:
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            df = yf.download(ticker, progress=False, multi_level_index=False, auto_adjust=False, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao baixar %s (tentativa %d): %s", ticker, tentativa, exc)
            df = pd.DataFrame()
        if not df.empty:
            return df.dropna(subset=["Close"])
        if tentativa < TENTATIVAS:
            time.sleep(ESPERA_S)
    logger.error("Sem dados para %s (%s)", ticker, kwargs)
    return pd.DataFrame()


def dia_util_anterior(dia: pd.Timestamp) -> pd.Timestamp:
    return dia - pd.offsets.BDay(1)


def formatar(df: pd.DataFrame, ativo: str, ticker: str, tipo: str, duracao, fonte: str,
             agora: pd.Timestamp) -> pd.DataFrame:
    """Converte o DataFrame do Yahoo para o formato longo do snapshot."""
    idx = pd.DatetimeIndex(df.index)
    if tipo == "1d":
        # barra diaria (indice ja normalizado para datas): inicio/fim ficam como a data
        inicio = fim = idx.strftime("%Y-%m-%d")
    else:
        ini = idx.tz_convert(TZ)
        inicio = ini.strftime("%Y-%m-%d %H:%M:%S%z")
        fim = (ini + duracao).strftime("%Y-%m-%d %H:%M:%S%z")
    return pd.DataFrame({
        "ativo": ativo, "ticker": ticker, "tipo": tipo, "inicio": inicio, "fim": fim,
        "open": df["Open"].to_numpy(), "high": df["High"].to_numpy(), "low": df["Low"].to_numpy(),
        "close": df["Close"].to_numpy(), "volume": df["Volume"].to_numpy(),
        "fonte": fonte, "coletado_em": agora.strftime("%Y-%m-%d %H:%M:%S%z"),
    })[COLUNAS]


def diario_ate_ontem(ticker: str, hoje: pd.Timestamp) -> pd.DataFrame:
    """Barras diarias com data < hoje (a barra de hoje traz o preco do momento)."""
    d = baixar(ticker, period="10d", interval="1d")
    if d.empty:
        return d
    datas = pd.DatetimeIndex(d.index)
    datas = (datas.tz_convert(TZ).tz_localize(None) if datas.tz is not None else datas).normalize()
    d.index = datas
    return d[d.index < hoje.tz_localize(None)]


def coletar_manha(agora: pd.Timestamp) -> tuple[pd.DataFrame, list]:
    hoje = agora.normalize()
    inicio_janela = dia_util_anterior(hoje) + INICIO_JANELA
    partes, falhas = [], []

    for ativo, ticker in ATIVOS_5M.items():
        h5 = baixar(ticker, period="5d", interval="5m", prepost=True)
        if h5.empty:
            falhas.append(ticker)
            continue
        ini = pd.DatetimeIndex(h5.index).tz_convert(TZ)
        h5 = h5[(ini >= inicio_janela) & (ini + DURACAO_5M <= agora)]   # so candles ja encerrados
        partes.append(formatar(h5, ativo, ticker, "5m", DURACAO_5M, "yahoo_5m_prepost", agora))
        ultimo = pd.DatetimeIndex(h5.index).tz_convert(TZ).max() + DURACAO_5M if len(h5) else None
        logger.info("  %-10s %-6s %4d candles de 5m | ultimo fim: %s", ativo, ticker, len(h5), ultimo)

    for ativo, ticker in [ALVO, *ATIVOS_5M.items()]:
        d = diario_ate_ontem(ticker, hoje)
        if d.empty:
            falhas.append(ticker + " (1d)")
            continue
        partes.append(formatar(d.tail(1), ativo, ticker, "1d", None, "yahoo_1d_ultimo_fechamento", agora))
        logger.info("  %-10s %-9s ultimo fechamento diario: %s = %.4f",
                    ativo, ticker, d.index[-1].date(), d["Close"].iloc[-1])
        if ativo == ALVO[0] and d.index[-1] != dia_util_anterior(hoje).tz_localize(None):
            logger.warning("  Ultimo fechamento do BOVA11 e de %s, nao do dia util anterior "
                           "(feriado ou falha do Yahoo)", d.index[-1].date())
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=COLUNAS), falhas


def coletar_noite(agora: pd.Timestamp) -> tuple[pd.DataFrame, list]:
    hoje = agora.normalize()
    ativo, ticker = ALVO

    # Candles de 5m do pregao de hoje (sempre gravados; tambem servem de reserva da barra diaria)
    h5 = baixar(ticker, period="5d", interval="5m")
    if not h5.empty:
        ini = pd.DatetimeIndex(h5.index).tz_convert(TZ)
        h5 = h5[(ini.normalize() == hoje) & (ini + DURACAO_5M <= agora)]
    partes = []
    if not h5.empty:
        partes.append(formatar(h5, ativo, ticker, "5m", DURACAO_5M, "yahoo_5m", agora))
        logger.info("  %s: %d candles de 5m no pregao de %s", ativo, len(h5), hoje.date())

    d = baixar(ticker, period="5d", interval="1d")
    if not d.empty:
        datas = pd.DatetimeIndex(d.index)
        datas = (datas.tz_convert(TZ).tz_localize(None) if datas.tz is not None else datas).normalize()
        d.index = datas
        d = d[d.index == hoje.tz_localize(None)]
    if not d.empty:
        logger.info("  %s %s: O=%.2f H=%.2f L=%.2f C=%.2f (barra diaria)", ativo, hoje.date(),
                    *d[["Open", "High", "Low", "Close"]].iloc[0])
        partes.insert(0, formatar(d, ativo, ticker, "1d", None, "yahoo_1d", agora))
    elif not h5.empty:
        # Sem barra diaria de hoje: monta o candle com os 5m do pregao regular
        agregado = pd.DataFrame({"Open": [h5["Open"].iloc[0]], "High": [h5["High"].max()],
                                 "Low": [h5["Low"].min()], "Close": [h5["Close"].iloc[-1]],
                                 "Volume": [h5["Volume"].sum()]}, index=[hoje.tz_localize(None)])
        logger.warning("  %s: sem barra diaria de hoje; candle montado com %d candles de 5m", ativo, len(h5))
        partes.insert(0, formatar(agregado, ativo, ticker, "1d", None, "agregado_5m", agora))
    else:
        logger.warning("  %s: sem barra diaria nem candles de 5m em %s (sem pregao?)", ativo, hoje.date())
        return pd.DataFrame(columns=COLUNAS), [ticker + " (sem pregao)"]
    return pd.concat(partes, ignore_index=True), []


def caminho_livre(pasta: Path, dia: pd.Timestamp, sufixo: str, agora: pd.Timestamp) -> Path:
    alvo = pasta / f"{dia:%Y-%m-%d}_{sufixo}.csv"
    if alvo.exists():
        alvo = pasta / f"{dia:%Y-%m-%d}_{sufixo}_{agora:%H%M%S}.csv"
        logger.warning("Arquivo do dia ja existe; gravando em %s", alvo.name)
    return alvo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modo", choices=["manha", "noite"])
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    args = parser.parse_args()

    args.saida.mkdir(parents=True, exist_ok=True)
    configurar_log(args.saida)
    agora = pd.Timestamp.now(tz=TZ)
    logger.info("Coleta '%s' iniciada em %s", args.modo, agora)

    if args.modo == "manha":
        dados, falhas = coletar_manha(agora)
        sufixo = "9h15"
    else:
        dados, falhas = coletar_noite(agora)
        sufixo = "19h"

    if not dados.empty:
        arquivo = caminho_livre(args.saida, agora, sufixo, agora)
        with open(arquivo, "x", newline="", encoding="utf-8") as f:   # "x": nunca sobrescreve
            dados.to_csv(f, index=False, float_format="%.6f")
        logger.info("Salvo: %s (%d linhas)", arquivo, len(dados))
    if falhas:
        logger.error("Falhas: %s", ", ".join(falhas))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
