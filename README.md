# IOSI: apoio à decisão intradiária para venda de straddle no BOVA11

> Sistema de ponta a ponta, da coleta de dados ao alerta no Telegram e ao dashboard no Supabase, que estima **quanto o BOVA11 vai oscilar no dia** e **se o dia tende a ser lateral**. Cada previsão é conferida automaticamente contra o resultado real.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-dados-150458?logo=pandas&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-Postgres-3FCF8E?logo=supabase&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-alertas-26A5E4?logo=telegram&logoColor=white)

---

## Resumo 

| | |
|---|---|
| **Problema** | Quem vende *straddle* ganha quando o dia é lateral e perde quando o mercado dispara. A decisão precisa ser tomada logo após a abertura, com pouca informação. |
| **Solução** | Um pipeline automatizado que às **10h30** publica a amplitude esperada do dia, uma faixa provável de preço e a probabilidade de dia lateral, e às **19h** confere o que aconteceu. |
| **Dados** | 17 anos de pregões (2008–2026) de BOVA11, EWZ, futuros do S&P 500 e do Brent e VIX, além de candles intradiários de 1h e 5 min. |
| **Validação** | *Walk-forward* mensal sem vazamento de informação: **2.649 previsões fora da amostra em 140 meses**, sempre comparadas a baselines ingênuos. |
| **Resultado principal** | A faixa provável cobriu a máxima **e** a mínima do dia em **80,9%** dos pregões testados (meta: 80%). |
| **Stack** | Python, pandas, scikit-learn, yfinance, Supabase (Postgres + RLS), GitHub Actions, Telegram Bot, HTML/JS. |

### Competências demonstradas

- **Ciência de dados aplicada:** seleção de variáveis por correlação simples e parcial, comparação de 30+ modelos candidatos, calibração de intervalos por quantis empíricos.
- **Rigor estatístico:** validação temporal (*walk-forward*), regras *point-in-time* cobertas por testes automatizados, baselines obrigatórios e testes de robustez a outliers.
- **Engenharia de dados:** ingestão de várias fontes, tratamento de barras inválidas, calendário oficial da B3, alinhamento de fusos horários.
- **Backend e banco de dados:** modelagem no Postgres com colunas geradas, views de KPI e Row Level Security no Supabase.
- **MLOps / automação:** execução agendada no GitHub Actions, log de execuções, monitoramento da saúde do pipeline e validação automática das previsões.
- **Comunicação técnica:** metodologia documentada como artigo científico, com revisão da literatura e limitações declaradas.

---

## Como funciona

```
                ┌──────────────────────────────────────────────────────────┐
  yfinance ───► │  Coleta (market_data.py)                                 │
  (diário,      │  BOVA11 · EWZ · ES=F · ^VIX · BZ=F                       │
   1h, 5 min)   └───────────────┬──────────────────────────────────────────┘
                                ▼
                ┌──────────────────────────────────────────────────────────┐
                │  Features point-in-time (HAR 1/5/22 dias, VIX t−1, ...)  │
                └───────────────┬──────────────────────────────────────────┘
                                ▼
   10h30  ┌──────────────────────────────┐        ┌────────────────────────┐
  ───────►│  predictor_1030.py           │───────►│  Card no Telegram      │
          │  amplitude · faixa · lateral │        └────────────────────────┘
          └──────────────┬───────────────┘
                         ▼
   19h00  ┌──────────────────────────────┐        ┌────────────────────────┐
  ───────►│  validador_19h.py            │───────►│  Supabase (Postgres)   │
          │  erro, Brier, faixa acertou? │        │  tabelas + views KPI   │
          └──────────────────────────────┘        └───────────┬────────────┘
                                                              ▼
                                                  ┌────────────────────────┐
                                                  │  Dashboard (HTML/JS)   │
                                                  └────────────────────────┘
```

### O que o sistema entrega às 10h30

| Saída | Modelo | Por que esse modelo |
|---|---|---|
| **Amplitude esperada do dia** (máxima − mínima, em % e em R$) | Regressão linear em log com HAR 1/5/22 dias + VIX (`HAR+VIX_ols_log`) | Menor MAE entre 8 candidatos, inclusive Random Forest e Gradient Boosting |
| **Faixa provável até o fechamento** | Multiplicadores empíricos calibrados sobre as previsões fora da amostra | Cobertura de máxima e mínima medida, não suposta |
| **Probabilidade de dia lateral** (\|fechamento − abertura\| < 1%) | Regressão logística L2, janela móvel de 5 anos | Menor Brier entre 9 candidatos |

> **Decisão de projeto:** o sistema **não prevê a direção** do fechamento. No teste fora da amostra nenhum modelo superou o palpite "retorno = 0" no retorno intradiário, então essa saída foi retirada em vez de exibir um número sem valor preditivo.

---

## Resultados (validação fora da amostra)

Os números abaixo vêm do *walk-forward* mensal: a cada mês, o modelo é treinado só com os dados anteriores e prevê o mês seguinte, sem embaralhamento.

| Alvo | Métrica | Modelo | Baseline | Ganho |
|---|---|---|---|---|
| Amplitude (n = 2.649) | MAE (pp) | **0,4870** | 0,5123 (média de 5 dias) | **+4,9%** |
| Amplitude | MAE (pp) | **0,4870** | 0,5308 (média de 22 dias) | **+8,3%** |
| Amplitude | Correlação de Spearman | **0,578** | 0,547 | — |
| Dia lateral (n = 2.549) | Brier | **0,2061** | 0,2131 (taxa dos últimos 252 dias) | **+3,3%** |
| Dia lateral | AUC | **0,647** | 0,587 | — |
| Gap de abertura (n = 260) | MAE (pp) | **0,4775** | 0,5487 (gap = 0) | **+13,0%** |
| Gap de abertura | Acerto de direção | **68,8%** | 50% | — |
| Faixa provável (n = 2.384) | Máx. e mín. dentro | **80,9%** | meta de 80% | ✓ |

**Como as variáveis foram escolhidas.** Foram testados 15+ ativos (índices, futuros, moedas). O câmbio (USD/BRL, DXY) parecia relevante, mas a **correlação parcial** mostrou que ele não acrescenta nada depois de controlar pelo EWZ: nenhuma das 16 parciais foi significativa. Por isso foi removido.

---

## Destaques de engenharia

- **Zero vazamento de informação.** Uma barra diária de data D só é usada a partir de D+1, e um candle intradiário só depois de fechado. Essas regras são verificadas por testes automatizados (`tests/test_dataset_9h15.py`, `tests/test_1030_point_in_time.py`).
- **O banco calcula as métricas.** No Supabase, erros, acertos de faixa e de direção são **colunas geradas** do Postgres, então nunca ficam inconsistentes com os dados brutos. Views (`vw_kpis_gerais`, entre outras) alimentam o dashboard.
- **Segurança.** RLS no Supabase: o backend grava com a chave secreta e o dashboard só lê com a chave pública. Segredos ficam no `.env` local e nos *secrets* do GitHub Actions.
- **Auditabilidade.** Cada previsão grava as features exatas usadas, o que permite reproduzir qualquer resultado.
- **Observabilidade.** A tabela `execucoes` registra tipo, duração, status, traceback e origem (local ou GitHub Actions) de cada execução.
- **Bugs reais encontrados e corrigidos**, entre eles:
  - a previsão ignorava o último fechamento (features deslocadas em um pregão);
  - no GitHub Actions (UTC), a data do pregão ficava 3 horas adiantada, corrigida com `America/Sao_Paulo`;
  - em feriados, a conferência validava o candle de outro dia; agora marca `SEM_DADOS`.

---

## Evolução do projeto

1. **Versão 1:** quatro modelos (XGBoost, LightGBM, Ridge, Random Forest) previam o preço de fechamento. O dashboard passou a mostrar **cada métrica ao lado de um baseline ingênuo**, e isso deixou claro que a "assertividade" de ~99% era ilusória: o BOVA11 varia cerca de 1% ao dia, então até "amanhã = hoje" acerta 99%.
2. **Versão 2 (atual):** o foco mudou para o que interessa ao *straddle*, que é **volatilidade e lateralidade** em vez de direção, com validação *walk-forward*, seleção formal de variáveis e uma faixa calibrada.

Essa mudança é, para mim, a principal lição do projeto: **medir contra o baseline certo muda as conclusões.**

---

## Limitações (declaradas com transparência)

- Nos **últimos 2 anos**, o modelo de dia lateral ficou ligeiramente **pior** que o baseline (−0,18%), e a cobertura da faixa caiu para 77,6%.
- O teste do modelo de gap tem só 13 meses e inclui um dia atípico (05/10/2026, pós-eleição). Sem ele, o R² fora da amostra cai de 0,42 para 0,23.
- O modelo vencedor foi escolhido no mesmo período em que é avaliado, sem um conjunto de validação separado.
- Dados do Yahoo Finance têm atraso de ~15 min na B3, histórico de 5 min de apenas 60 dias e barras inválidas ocasionais.
- Não há avaliação financeira do *straddle* (prêmio, custos, P&L): o sistema é de **apoio à decisão**, não uma estratégia de trading.

---

## Como rodar

```bash
# 1. Dependências
pip install -r requirements.txt

# 2. Credenciais (opcional: sem elas, o sistema grava só localmente)
cp .env.example .env
#    preencha SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY

# 3. Banco: no SQL Editor do Supabase, execute supabase/schema.sql

# 4. Pipeline
python src/market_data.py              # coleta o histórico
python src/treino_walkforward.py       # walk-forward e escolha dos modelos
python src/calibrar_faixa.py           # calibra a faixa provável
python src/predictor_1030.py           # previsão do dia
python src/validador_19h.py            # conferência com o resultado real

# 5. Testes
pytest tests/
```

**Dashboard:** abra `dashboard/index.html`, clique em *Conectar ao Supabase* e informe a URL e a chave pública. Sem conexão, ele exibe dados de demonstração.

**Automação:** o workflow em `.github/workflows/` executa a previsão e a validação nos horários de pregão. Basta cadastrar `SUPABASE_URL` e `SUPABASE_SERVICE_ROLE_KEY` em *Settings → Secrets and variables → Actions*.

---

## Estrutura

```
src/                 coleta, features, treino walk-forward, previsão e validação
models/              modelos treinados (.joblib) e metadados (vencedores.json, faixa.json)
relatorios/          métricas do walk-forward, correlações e resumos de cada fase
data/                bases de treino, snapshots intradiários e histórico de previsões
tests/               testes das regras point-in-time
supabase/schema.sql  tabelas, views de KPI e políticas RLS
dashboard/           painel estático (HTML/JS) conectado ao Supabase
.github/workflows/   execução agendada
```

---

## Base teórica

O modelo de amplitude parte do **HAR de Corsi (2009)**, referência em previsão de volatilidade. O protocolo de validação segue **Cerqueira, Torgo e Mozetič (2020)** sobre avaliação de séries temporais e considera o risco de sobreajuste descrito por **Bailey *et al.* (2017)**. A revisão completa e a metodologia estão documentadas no artigo que acompanha o projeto.

---

## Autor

**Murilo Schittino**: [[LinkedIn](https://www.linkedin.com/in/murilo-schittino/)](#) · [muriloschittino@gmail.com](#) · 

> *Aviso: projeto acadêmico e de portfólio. Nada aqui é recomendação de investimento.*
