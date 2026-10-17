# IOSI – Fase A: análise de correlação corrigida e completa

Gerado por `src/correlacao_ativos.py` em 16/10/2026. Os arquivos citados ficam nesta pasta.

**Convenções**

- Valores **entre parênteses** são insignificantes: |r| < 2/√n. O limiar é cerca de 0,06 com n ≈ 1.245 (5 anos), 0,09 com n ≈ 490 (2 anos) e 0,26 com n = 57 (5 minutos).
- **Pearson** é o padrão. **Spearman** aparece onde há valores extremos, porque o dia 05/10/2026 (primeiro pregão após a eleição, EWZ +12,6% no pré-mercado) distorce o Pearson.
- **Gap** é `open/close(t−1) − 1`. O retorno de fechamento é `close/close(t−1) − 1`. O retorno intradiário é `close/open − 1`. A amplitude é `(high − low)/open`.

---

## 1. O que mudou

| Item do roteiro | Como foi feito |
|---|---|
| 1. Local de trabalho | Por decisão posterior, todo o trabalho fica na pasta `IOSI-experimentos` (fora do repositório), com a mesma estrutura: o script em `src/correlacao_ativos.py` e os resultados em `relatorios/correlacao_ativos/`. Rodado a partir dessa pasta, com um .venv fora do OneDrive (`C:\Users\muril\.venvs\iosi-experimentos`). O resultado é idêntico ao da execução anterior no repositório (`saida_execucao.txt` × `saida_execucao_repo.txt`). |
| 2. Datas do câmbio, Etapa 1 | Recuo de **1 barra**, e não de 1 dia de calendário. O Yahoo não tem barra de sábado: o fechamento de sexta só aparece na barra de segunda, e recuar 1 dia de calendário perderia a sexta. A barra do dia corrente, com o preço do momento, é descartada. |
| 2. Datas do câmbio, Etapa 2 | O fechamento anterior das moedas é o candle de 1h que termina no fechamento da B3 em D−1. Esse horário vem do último candle de 1h do BOVA11 do dia (em todos os meses, o último candle começa às 16h e vai até as 17h). |
| 3. Novos alvos | Retorno intradiário, \|close − open\|/open e amplitude (high − low)/open. |
| 4. Novas features | Nível do VIX em t−1, amplitude média do BOVA11 em 1, 5 e 22 dias, horário do último negócio do EWZ antes das 9h, e EWZ às 9h × 9h15. |
| 5. Correlação parcial | USDBRL e DXY, controlando pelo EWZ e também por EWZ + ES=F + amplitude média de 22 dias. |
| 6. Significância | Marcada nas tabelas, nos heatmaps e na coluna `significativo` dos CSVs. |

**A correção do câmbio funcionou.** A matriz da Etapa 1 agora faz sentido econômico:

| Par | Antes | Depois |
|---|---|---|
| DXY × EURUSD | −0,12 | **−0,87** |
| DXY × USDJPY | 0,05 | 0,59 |
| BOVA11 × USDBRL | 0,03 | **−0,58** |
| EWZ × USDBRL | −0,01 | −0,79 |

---

## 2. Etapa 1 – diário, 5 anos (n ≈ 1.245)

| Ativo | ret × ret BOVA (t) | ret × ret BOVA (t−1) | \|ret\| × \|ret\| BOVA (t) | \|ret\| × \|ret\| BOVA (t−1) | \|ret\| × amp BOVA (t) | \|ret\| × amp BOVA (t−1) |
|---|---|---|---|---|---|---|
| BOVA11 | — | (−0,01) | — | (0,05) | — | 0,16 |
| IBOV | 0,98 | (−0,01) | 0,97 | (0,03) | 0,54 | 0,15 |
| EWZ | 0,88 | (−0,01) | 0,79 | (0,02) | 0,46 | 0,12 |
| SP500 | 0,38 | (−0,01) | 0,25 | 0,09 | 0,27 | 0,19 |
| NASDAQ | 0,33 | (−0,01) | 0,21 | 0,09 | 0,25 | 0,17 |
| RUSSELL2000 | 0,40 | (0,01) | 0,25 | (0,05) | 0,20 | 0,12 |
| VIX | −0,36 | (0,01) | 0,15 | (0,02) | 0,16 | 0,10 |
| USDBRL | −0,58 | (0,03) | 0,41 | (−0,00) | 0,29 | 0,12 |
| DXY | −0,22 | (0,02) | 0,14 | 0,09 | 0,19 | 0,14 |
| EURBRL | −0,41 | (0,03) | 0,28 | (0,03) | 0,24 | 0,18 |
| GBPBRL | −0,45 | (0,04) | 0,28 | (0,03) | 0,26 | 0,17 |
| USDJPY | −0,10 | (0,03) | 0,10 | (0,03) | 0,13 | 0,09 |
| EURUSD | 0,19 | (−0,02) | 0,10 | 0,09 | 0,17 | 0,14 |

(Nesta etapa, "amp BOVA" é `(high − low)/close(t−1)`, a mesma definição da primeira rodada.)

- **Com dados de t−1, a direção continua imprevisível:** todas as correlações defasadas são insignificantes, inclusive as de câmbio depois da correção.
- **O tamanho do movimento tem alguma persistência:** um \|ret\| alto em t−1 se associa a uma amplitude maior em t (de 0,09 a 0,19). As features próprias da seção 4 capturam isso melhor.

---

## 3. Etapa 2 – overnight até 9h BRT (~2 anos, n = 490)

O S&P 500, o Nasdaq, o Russell e o IBOV não têm pré-mercado no Yahoo: às 9h, o último preço deles tem cerca de 15h. Por isso ficaram de fora (ver `overnight_diagnostico.csv`).

**Pearson**

| Ativo | var × gap | var × ret fech. | var × ret intrad. | \|var\| × \|gap\| | \|var\| × \|ret fech.\| | \|var\| × \|close−open\|/open | \|var\| × amplitude |
|---|---|---|---|---|---|---|---|
| EWZ | 0,59 | 0,33 | (−0,04) | 0,62 | 0,38 | (0,01) | 0,17 |
| VIX | −0,28 | −0,12 | (0,07) | 0,17 | (0,03) | (0,06) | 0,11 |
| USDBRL | (−0,08) | (−0,03) | (0,02) | (0,09) | (−0,03) | (−0,02) | (0,02) |
| DXY | −0,12 | −0,14 | (−0,09) | 0,14 | (−0,04) | (−0,06) | (0,02) |
| EURBRL | (0,08) | 0,10 | (0,07) | 0,11 | (−0,03) | (−0,01) | (0,01) |
| GBPBRL | 0,15 | 0,14 | (0,06) | 0,13 | (−0,04) | (−0,07) | (−0,04) |
| USDJPY | (−0,08) | −0,14 | −0,12 | (0,09) | (−0,01) | (0,02) | (0,06) |
| EURUSD | 0,09 | 0,11 | (0,07) | 0,12 | (−0,02) | (−0,03) | (0,01) |
| ES=F | 0,32 | 0,11 | −0,12 | 0,18 | (0,03) | (0,04) | 0,16 |
| NQ=F | 0,32 | 0,10 | −0,12 | 0,16 | (0,01) | (0,03) | 0,12 |
| BZ=F | −0,15 | −0,11 | (−0,03) | 0,21 | 0,12 | (0,06) | 0,18 |

**Spearman (robusto a valores extremos)**

| Ativo | var × gap | var × ret fech. | var × ret intrad. | \|var\| × \|gap\| | \|var\| × \|ret fech.\| | \|var\| × \|close−open\|/open | \|var\| × amplitude |
|---|---|---|---|---|---|---|---|
| EWZ | 0,39 | 0,20 | (−0,05) | 0,28 | (0,09) | (0,03) | 0,14 |
| VIX | −0,26 | −0,13 | (0,01) | (0,03) | (−0,04) | (0,03) | (0,05) |
| USDBRL | (−0,05) | (−0,05) | (−0,02) | (0,01) | (−0,05) | (−0,05) | (0,03) |
| DXY | −0,18 | −0,18 | (−0,09) | (0,05) | (−0,06) | −0,10 | (−0,04) |
| EURBRL | 0,18 | 0,16 | (0,07) | (0,07) | (0,02) | (0,03) | (−0,01) |
| GBPBRL | 0,20 | 0,19 | (0,09) | (0,06) | (−0,02) | (−0,05) | (−0,09) |
| USDJPY | −0,12 | −0,17 | −0,11 | (0,03) | (0,01) | (0,02) | (0,02) |
| EURUSD | 0,18 | 0,17 | (0,07) | (0,05) | (−0,01) | (−0,01) | (−0,01) |
| ES=F | 0,30 | 0,12 | (−0,07) | (0,05) | (0,01) | (0,02) | (0,07) |
| NQ=F | 0,30 | 0,11 | (−0,08) | (0,06) | (0,01) | (0,00) | (0,06) |
| BZ=F | (−0,03) | (−0,06) | (−0,04) | 0,15 | (0,08) | (0,05) | 0,17 |

**Sem o dia 05/10/2026 (Pearson)**

| Ativo | var × gap | \|var\| × \|gap\| | \|var\| × amplitude |
|---|---|---|---|
| EWZ | 0,45 | 0,38 | 0,17 |
| ES=F | 0,37 | 0,24 | 0,17 |
| VIX | −0,33 | 0,19 | 0,11 |
| BZ=F | −0,17 | 0,27 | 0,18 |

**Leitura**

- **O EWZ no pré-mercado continua sendo o melhor sinal para o gap, mas o r = 0,59 está inflado por um único dia.** O valor robusto fica entre 0,39 (Spearman) e 0,45 (Pearson sem 05/10). O ES=F vem logo atrás (0,30 a 0,37) e é mais estável.
- **Nada do overnight explica o \|close − open\|/open,** que é o movimento *depois* da abertura e o que mais importa para um straddle montado na abertura: tudo ficou insignificante. O overnight explica o gap, não o pregão.
- **Na amplitude, o overnight contribui pouco:** \|EWZ\|, \|ES=F\| e \|Brent\| ficam entre 0,16 e 0,18 no Pearson. Só o Brent se mantém no Spearman (0,17).

---

## 4. Features próprias: nível do VIX e amplitude média do BOVA11 (HAR)

Tudo calculado até t−1, ou seja, conhecido antes da abertura.

**Pearson**

| Feature | \|gap\| (2 anos) | \|ret fech.\| (2 anos) | \|close−open\|/open (2 anos) | amplitude (2 anos) | \|gap\| (5 anos) | \|ret fech.\| (5 anos) | \|close−open\|/open (5 anos) | amplitude (5 anos) |
|---|---|---|---|---|---|---|---|---|
| VIX nível t−1 | 0,10 | 0,11 | 0,12 | 0,28 | 0,22 | 0,18 | 0,19 | 0,38 |
| amp. média 1d | 0,23 | 0,16 | (0,07) | 0,28 | 0,25 | 0,18 | 0,14 | 0,37 |
| amp. média 5d | 0,20 | 0,13 | (0,06) | 0,31 | 0,25 | 0,23 | 0,21 | 0,47 |
| amp. média 22d | 0,23 | 0,12 | (0,05) | 0,25 | 0,22 | 0,22 | 0,21 | 0,44 |

**Spearman**

| Feature | \|gap\| (2 anos) | \|ret fech.\| (2 anos) | \|close−open\|/open (2 anos) | amplitude (2 anos) | \|gap\| (5 anos) | \|ret fech.\| (5 anos) | \|close−open\|/open (5 anos) | amplitude (5 anos) |
|---|---|---|---|---|---|---|---|---|
| VIX nível t−1 | (0,05) | (0,04) | (0,06) | 0,11 | 0,21 | 0,15 | 0,14 | 0,36 |
| amp. média 1d | 0,17 | (0,05) | (−0,01) | 0,17 | 0,21 | 0,12 | 0,08 | 0,33 |
| amp. média 5d | 0,18 | 0,09 | (0,04) | 0,29 | 0,25 | 0,21 | 0,18 | 0,48 |
| amp. média 22d | 0,18 | 0,10 | (0,05) | 0,28 | 0,23 | 0,20 | 0,19 | 0,47 |

- **A amplitude média de 5 e 22 dias é o melhor preditor de amplitude de toda a análise:** 0,44 a 0,48 em 5 anos, robusto no Spearman. É bem mais forte que qualquer variação overnight, e por isso o baseline HAR da Fase D vai ser difícil de bater.
- **O nível do VIX ajuda em 5 anos (0,36 a 0,38), mas enfraquece nos últimos 2 anos** (Spearman de apenas 0,11).
- **Para o \|close − open\|/open, só aparece sinal em 5 anos (cerca de 0,2),** vindo do HAR e do VIX. Em 2 anos é insignificante.

---

## 5. EWZ no pré-mercado: frescor do preço e 9h × 9h15

Fonte: candles de 5 minutos, que o Yahoo só guarda por cerca de 60 dias: 57 pregões, de 27/07 a 16/10/2026.

O Yahoo informa volume 0 em todo o pré-mercado, mas só cria o candle de 5 minutos quando há negócio. Por isso, o fim do último candle é usado como horário do último negócio.

| Medida | Valor |
|---|---|
| Atraso mediano do último negócio às 9h | 0 min |
| Atraso no percentil 90 | 12 min |
| Dias com negócio nos últimos 15 min antes das 9h | 91% |
| Dias com negócio na última hora | 100% |
| Correlação entre var 9h e var 9h15 | 0,94 |
| Diferença mediana entre o preço das 9h15 e das 9h | 46 bps |

Nos 2 anos com candles de 1h, 94% dos dias têm candle na última hora antes das 9h. Em cerca de 1,4% dos dias o último preço tem 12h ou mais (feriados nos EUA com pregão na B3); nesses dias a variação do EWZ deve ser tratada como ausente.

| EWZ | var × gap (Pearson) | var × gap (Spearman) | \|var\| × \|gap\| (Pearson) | \|var\| × \|gap\| (Spearman) |
|---|---|---|---|---|
| às 9h | 0,81 | 0,39 | 0,85 | (0,12) |
| às 9h15 | 0,88 | **0,67** | 0,90 | **0,44** |
| às 9h, sem 05/10 | 0,34 | — | (0,07) | — |
| às 9h15, sem 05/10 | 0,65 | — | 0,47 | — |

**O preço das 9h15 é claramente melhor que o das 9h:** em 15 minutos o pré-mercado se move bastante (46 bps na mediana), na direção da abertura da B3.

Ressalvas: são só 57 dias, num período eleitoral. Além disso, o histórico de 2 anos em candles de 1h **não tem como capturar as 9h15**, porque o candle das 9h às 10h só fecha depois da abertura da B3.

---

## 6. Correlação parcial: o câmbio acrescenta algo além do EWZ?

| Moeda | Relação | Simples | Controle: EWZ | Controle: EWZ + ES=F + amp. 22d |
|---|---|---|---|---|
| USDBRL | var × gap | (−0,08) | (−0,01), p = 0,87 | (−0,01), p = 0,91 |
| USDBRL | \|var\| × \|gap\| | (0,09) | (−0,00), p = 0,98 | (−0,01), p = 0,88 |
| USDBRL | \|var\| × \|close−open\|/open | (−0,02) | (−0,02), p = 0,68 | (−0,02), p = 0,66 |
| USDBRL | \|var\| × amplitude | (0,02) | (−0,00), p = 0,91 | (−0,01), p = 0,80 |
| DXY | var × gap | −0,12 | (−0,04), p = 0,36 | (−0,07), p = 0,15 |
| DXY | \|var\| × \|gap\| | 0,14 | (0,03), p = 0,50 | (0,03), p = 0,45 |
| DXY | \|var\| × \|close−open\|/open | (−0,06) | (−0,06), p = 0,18 | (−0,08), p = 0,08 |
| DXY | \|var\| × amplitude | (0,02) | (−0,01), p = 0,74 | (−0,06), p = 0,19 |

**Resposta: não.** Controlando pelo EWZ, toda a informação de USDBRL e DXY desaparece. O pouco que o DXY mostrava sozinho já está no EWZ, que é cotado em dólar.

O câmbio importa *no mesmo dia* (BOVA × USDBRL = −0,58), mas às 9h15 não acrescenta nada para prever o dia.

---

## 7. Lista final de features recomendada (para a Fase B e a Fase C)

| Grupo | Feature (point-in-time, até 9h15 de t) | Alvo principal | Evidência |
|---|---|---|---|
| ✅ BOVA11 próprio | Amplitude média de 1, 5 e 22 dias (HAR) | Amplitude, \|gap\| | 0,44 a 0,48 em 5 anos; o melhor sinal de tamanho |
| ✅ VIX | Nível em t−1 | Amplitude | 0,36 a 0,38 em 5 anos (mais fraco nos últimos 2) |
| ✅ VIX | Variação overnight | Gap | −0,26 a −0,33, robusto |
| ✅ EWZ pré-mercado | Variação até **9h15** e seu valor absoluto, mais o atraso do último negócio | Gap, \|gap\| | 0,39 a 0,45 (9h, 2 anos) e até 0,67 (9h15, 57 dias) |
| ✅ ES=F | Variação overnight | Gap | 0,30 a 0,37, robusto |
| 🧪 BZ=F (Brent) | \|variação overnight\| | Amplitude, \|gap\| | 0,17 a 0,18 em amplitude, robusto no Spearman; levar à ablação da Fase D |
| ❌ NQ=F | — | — | Redundante com o ES=F (correlações idênticas) |
| ❌ USDBRL, DXY | — | — | Correlação parcial ≈ 0 depois do EWZ (p > 0,15) |
| ❌ EURBRL, GBPBRL, USDJPY, EURUSD | — | — | Fracas e redundantes com o EWZ e o DXY |
| ❌ ^BVSP, ^GSPC, ^IXIC, ^RUT à vista | — | — | Sem preço às 9h; o ES=F substitui |

Em relação à decisão inicial da Fase B, a mudança é: **USDBRL e DXY passam de 🧪 para ❌.** O resto se confirma.

**Implicações para as próximas fases**

1. **O \|close − open\|/open (o movimento do pregão em si) quase não é previsível com o overnight.** O peso da previsão de amplitude e de dia lateral vai estar no HAR e no VIX. O EWZ e o ES=F servem principalmente para o gap.
2. **Fase C:** o histórico em candles de 1h só dá o preço das 9h. Para usar as 9h15 é preciso começar já a gravar snapshots ao vivo às 9h15 (ou usar candles de 5 minutos, que cobrem só cerca de 60 dias). Até lá, treinar com as 9h e registrar a diferença.
3. **Fase D:** o dia pós-eleição (05/10/2026) domina várias métricas. Vale reportar os resultados também sem esse dia, ou usar uma perda robusta, para o modelo não ser escolhido por um único ponto.
4. **Câmbio no Yahoo:** qualquer código novo que use os tickers `=X` diários precisa da mesma correção de datas (`corrigir_datas_cambio`).

---

## 8. Arquivos

| Arquivo | Conteúdo |
|---|---|
| `correlacao_diaria_bova11.csv` | Etapa 1: Pearson, p-valor, Spearman, n e `significativo` |
| `matriz_correlacao_retornos.csv` | Etapa 1: matriz entre os ativos |
| `correlacao_overnight_bova11.csv` | Etapa 2: correlações overnight |
| `correlacao_features_proprias.csv` | VIX nível t−1 e HAR, em 2 e 5 anos |
| `correlacao_parcial_cambio.csv` | Correlação parcial de USDBRL e DXY |
| `ewz_5m_9h_vs_9h15.csv` | EWZ em 5 minutos: atraso do último negócio e var às 9h e às 9h15, por pregão |
| `overnight_dados.csv` | Base por pregão: variações overnight, features e alvos |
| `overnight_diagnostico.csv` | Frescor do preço às 9h e origem do fechamento anterior, por ativo |
| `heatmap_*.png` | Heatmaps (entre parênteses = insignificante) |
