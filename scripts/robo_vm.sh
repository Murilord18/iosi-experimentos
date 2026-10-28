#!/usr/bin/env bash
# robo_vm.sh
# Robo IOSI na VM (Linux), mesmo fluxo do .github/workflows/robo_iosi.yml:
#   manha      (09h17 BRT): coleta do pre-mercado + previsao da abertura (predictor_9h15.py)
#   manha_1030 (10h30 BRT): coleta com a B3 aberta + previsao das 10h30 (predictor_1030.py)
#   noite      (19h00 BRT): coleta + validacao
# Cada execucao: git pull, coleta, previsao/validacao, commit e push de data/.
# Um processo novo por execucao (nunca importar o modelo dentro de um processo que
# fica rodando: a data do dia e calculada na chamada, mas os dados baixados nao).
#
# Uso:   scripts/robo_vm.sh manha|manha_1030|noite
# Cron (VM em UTC; 12:17 UTC = 09:17 BRT, 13:30 UTC = 10:30 BRT, 22:00 UTC = 19:00 BRT):
#   17 12 * * 1-5  /home/<usuario>/iosi-experimentos/scripts/robo_vm.sh manha
#   30 13 * * 1-5  /home/<usuario>/iosi-experimentos/scripts/robo_vm.sh manha_1030
#   0  22 * * 1-5  /home/<usuario>/iosi-experimentos/scripts/robo_vm.sh noite
#
# Saidas fora do git (logs/, no .gitignore):
#   logs/robo_vm.log          log de todas as execucoes
#   logs/status_<modo>.txt    uma linha da ultima execucao do modo, para o bot ler:
#                             "<data> <modo> coleta=<ok|falha> etapa2=<ok|falha|-> push=<ok|falha|nada>"
# Codigo de saida: 0 se tudo deu certo, 1 se alguma etapa falhou (como o job do Actions).

set -uo pipefail

MODO="${1:-}"
case "$MODO" in
    manha|manha_1030|noite) ;;
    *) echo "uso: $0 manha|manha_1030|noite" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${IOSI_PYTHON:-$REPO/.venv/bin/python}"
LOGS="$REPO/logs"
mkdir -p "$LOGS"
exec >>"$LOGS/robo_vm.log" 2>&1

export TZ=America/Sao_Paulo
DATA="$(date +%Y-%m-%d)"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') $MODO ====="

# uma execucao por vez (o concurrency do workflow); a 10h30 pode levar ate 25 min
exec 9>"$LOGS/robo_vm.lock"
flock 9

cd "$REPO" || exit 1
# codigo e dados atualizados (inclusive o que o Actions tiver gravado como reserva)
git pull --rebase --autostash -q origin main || echo "aviso: git pull falhou; seguindo com o codigo local"

# o coletor das 10h30 pode esperar ate 25 min pelo candle do BOVA11 (atraso do Yahoo)
if timeout 40m "$PYTHON" src/coletor_snapshots.py "$MODO"; then coleta=ok; else coleta=falha; fi

# roda mesmo se a coleta falhar: o preditor busca os candles de 5m no Yahoo (fim <= 10h30)
# e o validador grava "sem_candle_real".
etapa2=-
case "$MODO" in
    manha)      if "$PYTHON" src/predictor_9h15.py; then etapa2=ok; else etapa2=falha; fi ;;
    manha_1030) if "$PYTHON" src/predictor_1030.py; then etapa2=ok; else etapa2=falha; fi ;;
    noite)      if "$PYTHON" src/validador_19h.py;  then etapa2=ok; else etapa2=falha; fi ;;
esac

push=nada
git add data/
if ! git diff --cached --quiet; then
    git -c user.name="robo-iosi-vm" -c user.email="robo-iosi-vm@users.noreply.github.com" \
        commit -q -m "robô: $DATA $MODO [skip ci]"
    push=falha
    for tentativa in 1 2 3; do
        if git pull --rebase -q origin main && git push -q origin HEAD:main; then push=ok; break; fi
        sleep $((tentativa * 20))
    done
fi

echo "$DATA $MODO coleta=$coleta etapa2=$etapa2 push=$push" | tee "$LOGS/status_$MODO.txt"
[ "$coleta" = ok ] && [ "$etapa2" != falha ] && [ "$push" != falha ]
