"""
carga_inicial_supabase.py
=========================
Projeto IOSI - envia uma vez para o Supabase o que ja existe em data/:
    data/snapshots/*.csv           -> snapshots
    data/previsoes/AAAA-MM-DD.json      -> previsoes (so a primeira previsao de cada dia: 9h15
    data/previsoes/AAAA-MM-DD_1030.json    e 10h30; no mesmo dia, a das 10h30 vai por ultimo)
    data/historico_previsoes.csv   -> previsoes (conferencia das 19h)

Usa os mesmos upserts do robo (storage_supabase.py): rodar de novo nao duplica nada.
Arquivos de data/simulacao/ nunca sao enviados.

Precisa de SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY no ambiente (no GitHub: Run workflow
com modo = carga_inicial). Termina com codigo 1 se algum envio falhar.

Uso:
    python src/carga_inicial_supabase.py
"""

import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

import storage_supabase as sb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("carga_inicial")

DATA = Path(__file__).parent.parent / "data"


def carregar_snapshots(pasta: Path) -> pd.DataFrame:
    # ordem do nome: o CSV coletado depois fica por ultimo e prevalece na mesma chave
    partes = []
    for arq in sorted(pasta.glob("*.csv")):
        df = pd.read_csv(arq, dtype={"inicio": str, "fim": str})
        df["arquivo"] = arq.name
        partes.append(df)
        logger.info("  %s: %d linhas", arq.name, len(df))
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def main() -> int:
    if not sb.configurado():
        logger.error("SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY precisam estar no ambiente")
        return 1
    ok = True

    logger.info("Snapshots (%s)", DATA / "snapshots")
    snaps = carregar_snapshots(DATA / "snapshots")
    if len(snaps):
        ok &= sb.upsert_snapshots(snaps)

    previsoes = sorted(p for p in (DATA / "previsoes").glob("*.json")
                       if re.fullmatch(r"\d{4}-\d{2}-\d{2}(_1030)?\.json", p.name))
    logger.info("Previsoes: %d arquivos", len(previsoes))
    for arq in previsoes:
        p = json.loads(arq.read_text(encoding="utf-8"))
        if p.get("simulacao"):
            continue
        ok &= sb.upsert_previsao(p, arq.name)

    hist = DATA / "historico_previsoes.csv"
    if hist.exists():
        h = pd.read_csv(hist)
        logger.info("Historico: %d linhas", len(h))
        ok &= sb.upsert_historico(h)

    logger.info("Carga inicial %s", "concluida" if ok else "COM FALHAS (ver avisos acima)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
