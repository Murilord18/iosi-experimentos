# Pesquisa: os primeiros 30 minutos do pregão ajudam a prever o resto do dia?

Gerado por `src/pesquisa_1030.py` em 18/10/2026 22:41. O modelo de produção **não** foi trocado: aqui só há resultados.

## a) Candles de 5 min (point-in-time às 10h30)

- Amostra: 59 pregões com candles de 5 min do Yahoo (23/07/2026 a 16/10/2026). Todas as features usam só candles com fim ≤ 10h30 (`leitura_1030.py`, o mesmo código do `predictor_1030.py`, coberto pelo teste point-in-time).
- No Yahoo, o 1º candle de 5 min do BOVA11 costuma ser o das 10h05 (o leilão de abertura do ETF termina por volta de 10h08). Assim, os "primeiros 30 min" são, na prática, ~22 min de negociação (mediana de 5 candles).
- Resto do dia: candles que começam depois do preço das 10h30 (até 16h55; o leilão de fechamento não está nos candles de 5 min, mas o close é o da barra diária).
- Correlação parcial: resíduo da feature e do alvo (log para a amplitude) depois de regredir os dois em log(HAR 1/5/22d) e log(VIX). Mede o que a feature acrescenta ao modelo atual.

**Alvo: amp_resto** (n = 59; insignificante se |r| < 2/√n ≈ 0,26, entre parênteses)

| Feature | Pearson | Spearman | Pearson sem 05/10 | Parcial (controle HAR+VIX) |
|---|---|---|---|---|
| amplitude dos primeiros 30 min | (0,15) | (0,06) | (0,13) | (0,06) |
| \|retorno dos primeiros 30 min\| | (0,10) | (0,03) | (0,12) | (0,06) |
| retorno dos primeiros 30 min | (-0,05) | (-0,02) | (-0,04) | (-0,03) |
| \|gap real\| | (0,25) | (0,19) | (0,19) | (0,20) |
| gap real | (0,25) | (0,16) | (0,17) | (0,21) |
| volume 10h-10h30 / média 20 pregões | 0,28 | (0,14) | (0,22) | (0,15) |
| \|variação EWZ até 10h30\| | (0,23) | (0,16) | (0,16) | (0,17) |
| variação EWZ até 10h30 | (0,20) | (0,07) | (0,09) | (0,17) |
| \|variação ES=F até 10h30\| | (0,07) | (0,03) | (0,10) | (-0,06) |
| variação ES=F até 10h30 | (0,04) | (-0,04) | (0,05) | (-0,16) |
| variação VIX até 10h30 | (-0,06) | (0,03) | (-0,08) | (0,03) |
| referência: HAR 5d | (0,09) | (0,12) | (0,06) | — |
| referência: HAR 22d | (0,19) | (0,16) | (0,17) | — |
| referência: nível do VIX (t−1) | (-0,03) | (0,01) | (-0,02) | — |

**Alvo: |ret_resto|** (n = 59; insignificante se |r| < 2/√n ≈ 0,26, entre parênteses)

| Feature | Pearson | Spearman | Pearson sem 05/10 | Parcial (controle HAR+VIX) |
|---|---|---|---|---|
| amplitude dos primeiros 30 min | (0,13) | (0,07) | (0,14) | (0,09) |
| \|retorno dos primeiros 30 min\| | (0,11) | (0,04) | (0,11) | (0,11) |
| retorno dos primeiros 30 min | (0,10) | (0,14) | (0,09) | (0,08) |
| \|gap real\| | (-0,03) | (0,02) | (0,06) | (-0,06) |
| gap real | (0,01) | (0,05) | (0,10) | (-0,03) |
| volume 10h-10h30 / média 20 pregões | (0,05) | (0,12) | (0,14) | (0,08) |
| \|variação EWZ até 10h30\| | (-0,07) | (-0,09) | (-0,02) | (-0,10) |
| variação EWZ até 10h30 | (-0,02) | (-0,01) | (0,06) | (-0,09) |
| \|variação ES=F até 10h30\| | (-0,10) | (0,04) | (-0,12) | (0,05) |
| variação ES=F até 10h30 | (-0,12) | (-0,11) | (-0,12) | (-0,16) |
| variação VIX até 10h30 | (0,01) | (0,03) | (0,02) | (0,06) |
| referência: HAR 5d | (-0,06) | (-0,09) | (-0,05) | — |
| referência: HAR 22d | (-0,09) | (-0,13) | (-0,08) | — |
| referência: nível do VIX (t−1) | (0,06) | (0,07) | (0,06) | — |

**Alvo: lateral_resto** (n = 59; insignificante se |r| < 2/√n ≈ 0,26, entre parênteses)

| Feature | Pearson | Spearman | Pearson sem 05/10 | Parcial (controle HAR+VIX) |
|---|---|---|---|---|
| amplitude dos primeiros 30 min | (-0,03) | (0,03) | (-0,04) | (0,03) |
| \|retorno dos primeiros 30 min\| | (-0,04) | (0,00) | (-0,03) | (-0,01) |
| retorno dos primeiros 30 min | (-0,13) | (-0,09) | (-0,13) | (-0,11) |
| \|gap real\| | (0,08) | (-0,06) | (-0,00) | (0,12) |
| gap real | (0,10) | (0,05) | (0,04) | (0,14) |
| volume 10h-10h30 / média 20 pregões | (-0,01) | (-0,09) | (-0,11) | (-0,03) |
| \|variação EWZ até 10h30\| | (0,10) | (0,08) | (0,05) | (0,15) |
| variação EWZ até 10h30 | (0,09) | (0,02) | (0,02) | (0,14) |
| \|variação ES=F até 10h30\| | (0,10) | (0,00) | (0,12) | (-0,02) |
| variação ES=F até 10h30 | (0,11) | (0,04) | (0,11) | (0,15) |
| variação VIX até 10h30 | (-0,00) | (-0,01) | (-0,01) | (-0,05) |
| referência: HAR 5d | (0,11) | (0,09) | (0,10) | — |
| referência: HAR 22d | (0,03) | (0,08) | (0,02) | — |
| referência: nível do VIX (t−1) | (-0,03) | (-0,02) | (-0,03) | — |

**Alvo: amplitude do dia** (n = 59; insignificante se |r| < 2/√n ≈ 0,26, entre parênteses)

| Feature | Pearson | Spearman | Pearson sem 05/10 | Parcial (controle HAR+VIX) |
|---|---|---|---|---|
| amplitude dos primeiros 30 min | 0,26 | (0,19) | (0,25) | (0,21) |
| \|retorno dos primeiros 30 min\| | (0,17) | (0,11) | (0,18) | (0,16) |
| retorno dos primeiros 30 min | (-0,03) | (-0,03) | (-0,02) | (-0,01) |
| \|gap real\| | (0,20) | (0,21) | (0,18) | (0,18) |
| gap real | (0,24) | (0,19) | (0,22) | (0,21) |
| volume 10h-10h30 / média 20 pregões | (0,23) | (0,15) | (0,20) | (0,17) |
| \|variação EWZ até 10h30\| | (0,16) | (0,14) | (0,09) | (0,14) |
| variação EWZ até 10h30 | (0,19) | (0,13) | (0,14) | (0,15) |
| \|variação ES=F até 10h30\| | (0,02) | (0,01) | (0,04) | (-0,11) |
| variação ES=F até 10h30 | (0,02) | (-0,06) | (0,03) | (-0,18) |
| variação VIX até 10h30 | (-0,08) | (0,00) | (-0,10) | (0,02) |
| referência: HAR 5d | (0,04) | (0,07) | (0,02) | — |
| referência: HAR 22d | (0,10) | (0,03) | (0,08) | — |
| referência: nível do VIX (t−1) | (0,00) | (0,04) | (0,01) | — |

## b) Proxy com 2 anos: candle de 1h das 10h–11h

> ⚠️ **Proxy:** o candle de 1h das 10h usa informação **até 11h**, não até 10h30. Por isso o "resto do dia" aqui começa às 11h (`amp_resto_proxy` = (máx. − mín. dos candles de 1h a partir de 11h) / close das 11h). A amplitude do dia inteiro inclui a 1ª hora, então o ganho para ela é em parte mecânico.

- Amostra: 490 pregões (21/10/2024 a 16/10/2026); walk-forward mensal, janela expansiva, mínimo de 120 pregões de treino; OLS em log.
- Baselines: `har_5d` e `har_22d` (para o resto do dia, multiplicados pela razão mediana alvo/har do treino, porque o resto do dia é menor que o dia inteiro).
- t do ganho: média da diferença diária de erro absoluto (HAR+VIX − modelo) dividida pelo erro padrão (ingênuo, sem correção de autocorrelação).

**Alvo: amp_resto (proxy, depois de 11h)** (n = 363, 2025-05-02 a 2026-10-16)

| Modelo | MAE (pp) | Spearman | Ganho vs HAR+VIX | t do ganho | Ganho vs har_5d_escalado | Ganho vs har_22d_escalado |
|---|---|---|---|---|---|---|
| HAR+VIX | 0,325 | 0,40 | 0,0% | — | 1,3% | -3,8% |
| HAR+VIX+amp_1h | 0,321 | 0,43 | 1,3% | 1,5 | 2,6% | -2,4% |
| har_5d_escalado | 0,329 | 0,43 | -1,3% | -0,6 | 0,0% | -5,1% |
| har_22d_escalado | 0,313 | 0,43 | 3,6% | 2,0 | 4,9% | 0,0% |

Sem 05/10/2026: ganho do HAR+VIX+amp_1h sobre o HAR+VIX = 1,2% (t = 1,3).

**Alvo: amplitude do dia** (n = 363, 2025-05-02 a 2026-10-16)

| Modelo | MAE (pp) | Spearman | Ganho vs HAR+VIX | t do ganho | Ganho vs har_5d | Ganho vs har_22d |
|---|---|---|---|---|---|---|
| HAR+VIX | 0,406 | 0,32 | 0,0% | — | 4,0% | 2,6% |
| HAR+VIX+amp_1h | 0,351 | 0,54 | 13,7% | 5,4 | 17,1% | 15,9% |
| har_5d | 0,423 | 0,36 | -4,2% | -1,2 | 0,0% | -1,5% |
| har_22d | 0,417 | 0,32 | -2,7% | -1,0 | 1,4% | 0,0% |

Sem 05/10/2026: ganho do HAR+VIX+amp_1h sobre o HAR+VIX = 13,2% (t = 5,3).

**Curva de aprendizado** (janela móvel de N pregões; mesmos dias de teste em todas as linhas)

| Alvo | Janela de treino | n teste | MAE HAR+VIX (pp) | MAE +amp_1h (pp) | Ganho |
|---|---|---|---|---|---|
| amp_resto_proxy | 60 | 233 | 0,421 | 0,414 | 1,5% |
| amp_resto_proxy | 120 | 233 | 0,382 | 0,382 | -0,2% |
| amp_resto_proxy | 250 | 233 | 0,356 | 0,352 | 1,0% |
| amp_resto_proxy | expansiva | 233 | 0,359 | 0,353 | 1,8% |
| amplitude | 60 | 233 | 0,507 | 0,480 | 5,3% |
| amplitude | 120 | 233 | 0,465 | 0,426 | 8,4% |
| amplitude | 250 | 233 | 0,457 | 0,403 | 11,9% |
| amplitude | expansiva | 233 | 0,454 | 0,396 | 12,9% |

Correlação parcial de log(amp_1h) com o alvo, controlando HAR+VIX (amostra toda): amp_resto proxy 0,16, amplitude do dia 0,43.

## c) Conclusão

- **Resto do dia (amp_resto): a informação do início do pregão quase não ajuda.** No proxy de 2 anos, acrescentar a amplitude da 1ª hora ao HAR+VIX reduz o MAE em 1,3% (t = 1,5, não significativo), e o modelo com a 1ª hora fica atrás do har_22d escalado (-2,4%). Com os 30 min reais (n = 59), nenhuma feature passa do limiar de 0,26 na correlação parcial com a amp_resto (amplitude dos 30 min: 0,06), e nem o HAR passa nessa amostra curta. Para |ret_resto| e lateral_resto, nada é significativo: a direção e o tamanho do movimento até o fechamento continuam imprevisíveis com o que se sabe às 10h30.
- **Dia inteiro (amplitude e faixa): ajuda bastante.** O ganho do proxy é de 13,7% (t = 5,4) e cresce com a janela de treino (5,3% com 60 pregões, 11,9% com 250, 12,9% com a janela expansiva). Parte é mecânica (a 1ª hora faz parte do dia), mas é exatamente o que interessa à faixa das 10h30, que precisa conter a máxima e a mínima do dia. Com os 30 min reais, a correlação parcial é 0,21 (n = 59, ainda abaixo do limiar), menor que a do proxy de 1h (0,43), como esperado para meia hora de pregão.
- **Quantos dias de snapshots das 10h30:** (1) para o efeito dos 30 min sobre a amplitude do dia ficar significativo (|r| > 2/√n), ≈ 91 pregões com r = 0,21; (2) o walk-forward mensal precisa de 120 pregões de treino antes do primeiro mês de teste, e a curva de aprendizado mostra o ganho ainda subindo de 120 para 250 pregões. **Recomendação: ~250 pregões (cerca de 1 ano) de snapshots das 10h30 para treinar e validar o modelo definitivo da amplitude do dia/faixa; com ~120 já dá para uma primeira avaliação.** Para a amp_resto, se o ganho do proxy (1,3%) for real, seriam necessários ≈ 655 dias só de teste para confirmá-lo (t = 2), o que não compensa: o melhor uso dos primeiros 30 min é na faixa do dia inteiro.
- Curva de aprendizado da amp_resto: 1,5% (60), -0,2% (120), 1,0% (250), 1,8% (expansiva), sem tendência clara (ruído).
- Os ~60 dias de candles de 5 min do Yahoo usados aqui (`dataset_5m_1030.csv`) adiantam o histórico: somados aos snapshots diários das 10h30, a contagem começa com eles. Como o Yahoo só guarda 60 dias, eles se perdem se não forem gravados: vale manter uma cópia deste CSV.

Arquivos: `dataset_5m_1030.csv`, `correlacoes_5m.csv`, `dataset_proxy_1h.csv`, `previsoes_proxy.csv`, `walkforward_proxy_metricas.csv`, `curva_aprendizado_proxy.csv`.
