"""
market_data.py
==============
Modulo 1 - Projeto IOSI (Indice de Oportunidade de Straddle Intraday)
Lista de ativos da Fase B (definida pelo resumo da Fase A, relatorios/correlacao_ativos).

Ativos Monitorados:
- BOVA11: Ativo alvo (ETF do Ibovespa)
- EWZ: ETF do Brasil em NY (pre-mercado = melhor sinal para o gap)
- ES_FUT: Futuro do S&P 500 (ES=F)
- VIX: Indice de volatilidade (^VIX), nivel em t-1 + variacao overnight
- BRENT_FUT: Futuro do Brent (BZ=F) - em teste, decidir na ablacao da Fase D

Removidos: USDBRL e DXY (correlacao parcial ~0 depois de controlar pelo EWZ),
USDJPY, GBPBRL, EURBRL, EURUSD, NQ=F (redundante com o ES=F) e indices a vista.

Autor: Patrick Campos (Modulo 1); lista de ativos alterada na Fase B (copia em IOSI-experimentos)
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ASSETS = {
    "BOVA11":    "BOVA11.SA",
    "EWZ":       "EWZ",
    "ES_FUT":    "ES=F",
    "VIX":       "^VIX",
    "BRENT_FUT": "BZ=F",       # em teste (ablacao da Fase D)
}

DB_PATH = Path(__file__).parent.parent / "market_data.db"
TABLE_NAME = "candles_daily"


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            asset     TEXT NOT NULL,
            date      TEXT NOT NULL,
            open      REAL NOT NULL,
            high      REAL NOT NULL,
            low       REAL NOT NULL,
            close     REAL NOT NULL,
            volume    REAL,
            timeframe TEXT NOT NULL DEFAULT '1d',
            PRIMARY KEY (asset, date, timeframe)
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_asset_date ON {TABLE_NAME}(asset, date)")
    conn.commit()
    return conn


def fetch_daily_candles(ticker: str, period: str = "5y") -> pd.DataFrame:
    raw = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise ValueError(f"Nenhum dado retornado para o ticker '{ticker}'.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def save_candles(conn: sqlite3.Connection, asset_name: str, df: pd.DataFrame, timeframe: str = "1d") -> int:
    rows = [
        (asset_name, row.date, row.open, row.high, row.low, row.close, row.volume, timeframe)
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        f"""INSERT OR REPLACE INTO {TABLE_NAME}
            (asset, date, open, high, low, close, volume, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def collect_asset(conn: sqlite3.Connection, asset_name: str, period: str = "5y") -> int:
    ticker = ASSETS.get(asset_name)
    if ticker is None:
        raise KeyError(f"Ativo '{asset_name}' nao cadastrado em ASSETS.")

    logger.info(f"Coletando {asset_name} ({ticker})...")
    df = fetch_daily_candles(ticker, period=period)
    qtd = save_candles(conn, asset_name, df)
    logger.info(f"{asset_name}: {qtd} candles salvos com sucesso.")
    return qtd


def collect_all(period: str = "5y", db_path: Path = DB_PATH) -> dict:
    conn = init_db(db_path)
    resultados = {}
    for asset_name in ASSETS:
        try:
            resultados[asset_name] = collect_asset(conn, asset_name, period=period)
        except Exception as e:
            logger.error(f"Falha ao coletar {asset_name}: {e}")
            resultados[asset_name] = 0
    conn.close()
    return resultados


def get_candles(asset_name: str, start: str = None, end: str = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = f"SELECT date, open, high, low, close, volume FROM {TABLE_NAME} WHERE asset = ?"
    params = [asset_name]

    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)

    query += " ORDER BY date ASC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def list_available_assets(db_path: Path = DB_PATH) -> list:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(f"SELECT DISTINCT asset FROM {TABLE_NAME} ORDER BY asset").fetchall()
    conn.close()
    return [r[0] for r in rows]


if __name__ == "__main__":
    print("Iniciando coleta do BOVA11, EWZ, ES=F, VIX e BZ=F...")
    resultados = collect_all(period="5y")
    print("\nResumo da Coleta:")
    for ativo, qtd in resultados.items():
        status = "OK" if qtd > 0 else "FALHOU"
        print(f"  {ativo:15s} -> {qtd:5d} candles  [{status}]")
