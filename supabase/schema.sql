-- =====================================================================
-- Projeto IOSI - tabelas do robo no Supabase
-- Colar no SQL Editor do Supabase e clicar em Run.
-- Pode ser rodado de novo sem erro (IF NOT EXISTS / DROP ... IF EXISTS).
--
-- O projeto foi criado com "Automatically expose new tables" DESATIVADO,
-- entao nenhuma role ganha acesso sozinha: os GRANTs abaixo sao explicitos.
--   - escrita (insert/update/delete) nas tres tabelas: so service_role (o robo)
--   - leitura de previsoes: anon (policy de SELECT)
--   - snapshots e execucoes: invisiveis para anon e authenticated
-- service_role ignora RLS (BYPASSRLS); anon depende da policy.
-- =====================================================================

-- ---------------------------------------------------------------------
-- snapshots: uma linha por candle dos CSVs data/snapshots/*.csv
-- inicio/fim ficam como texto, iguais ao CSV: data (AAAA-MM-DD) nas barras
-- diarias (tipo 1d) e horario com fuso (AAAA-MM-DD HH:MM:SS-0300) nos de 5m.
-- ---------------------------------------------------------------------
create table if not exists public.snapshots (
    id            bigint generated always as identity primary key,
    ativo         text not null,
    ticker        text not null,
    tipo          text not null,
    inicio        text not null,
    fim           text,
    open          double precision,
    high          double precision,
    low           double precision,
    close         double precision,
    volume        bigint,
    fonte         text,
    coletado_em   timestamptz,
    arquivo       text,                      -- CSV de origem (ex.: 2026-10-16_19h.csv)
    atualizado_em timestamptz not null default now(),
    constraint snapshots_ativo_tipo_inicio_key unique (ativo, tipo, inicio)
);

-- ---------------------------------------------------------------------
-- previsoes: uma linha por pregao. Junta a previsao das 9h15 (JSON de
-- data/previsoes/) e a conferencia das 19h (data/historico_previsoes.csv).
-- As colunas do historico mantem o nome do CSV; os campos do JSON que ja
-- existem no historico com outro nome vao para a coluna do historico
-- (gerado_em -> previsao_gerada_em, gap_previsto -> gap_prev,
-- abertura_esperada -> abertura_prev, amplitude_prevista -> amplitude_prev,
-- faixa_inferior -> faixa_inf, faixa_superior -> faixa_sup).
-- O status do JSON vira status_previsao; status e o do validador.
-- ---------------------------------------------------------------------
create table if not exists public.previsoes (
    data                      date primary key,
    -- historico_previsoes.csv
    status                    text,
    validado_em               timestamptz,
    simulacao                 boolean,
    previsao_gerada_em        timestamptz,
    fonte_gap_card            text,
    fonte_candle              text,
    fech_anterior             double precision,
    gap_prev_9h               double precision,
    gap_prev_9h15             double precision,
    gap_prev                  double precision,
    abertura_prev             double precision,
    amplitude_prev            double precision,
    faixa_inf                 double precision,
    faixa_sup                 double precision,
    prob_lateral              double precision,
    open                      double precision,
    high                      double precision,
    low                       double precision,
    close                     double precision,
    gap_real                  double precision,
    amplitude_real            double precision,
    abs_intraday_real         double precision,
    lateral_real              smallint,
    erro_gap_9h               double precision,
    erro_gap_9h15             double precision,
    erro_gap                  double precision,
    erro_abertura_rs          double precision,
    erro_amplitude            double precision,
    brier_lateral             double precision,
    max_dentro_faixa          smallint,
    min_dentro_faixa          smallint,
    amplitude_har_5d          double precision,
    erro_amplitude_har_5d     double precision,
    erro_gap_zero             double precision,
    taxa_lateral_252d         double precision,
    brier_lateral_taxa_252d   double precision,
    -- JSON da previsao (campos que nao estao no historico)
    status_previsao           text,
    data_fech_anterior        date,
    fonte_fech_anterior       text,
    amplitude_prevista_rs     double precision,
    faixa_k_inf               double precision,
    faixa_k_sup               double precision,
    faixa_cobertura_alvo      double precision,
    faixa_cobertura_card      double precision,
    baselines                 jsonb,
    modelos                   jsonb,
    features                  jsonb,
    overnight_9h              jsonb,
    overnight_9h15            jsonb,
    alerta_formatado_telegram text,
    arquivo_previsao          text,          -- JSON de origem (ex.: 2026-10-19.json)
    atualizado_em             timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- execucoes: uma linha por script rodado pelo robo (coleta, previsao,
-- validacao). Gravada como em_andamento no inicio e fechada no fim.
-- ---------------------------------------------------------------------
create table if not exists public.execucoes (
    id             bigint generated always as identity primary key,
    tipo           text not null check (tipo in ('manha', 'noite')),
    etapa          text not null check (etapa in ('coleta', 'previsao', 'validacao')),
    inicio         timestamptz not null default now(),
    fim            timestamptz,
    status         text not null default 'em_andamento'
                   check (status in ('em_andamento', 'sucesso', 'falha')),
    mensagem_erro  text,
    origem         text not null check (origem in ('github', 'local')),
    github_run_id  text
);

create index if not exists execucoes_inicio_idx on public.execucoes (inicio desc);

-- ---------------------------------------------------------------------
-- atualizado_em: renovado a cada update (o upsert do robo atualiza a linha)
-- ---------------------------------------------------------------------
create or replace function public.iosi_marcar_atualizado()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    new.atualizado_em := now();
    return new;
end;
$$;

drop trigger if exists snapshots_atualizado on public.snapshots;
create trigger snapshots_atualizado before update on public.snapshots
    for each row execute function public.iosi_marcar_atualizado();

drop trigger if exists previsoes_atualizado on public.previsoes;
create trigger previsoes_atualizado before update on public.previsoes
    for each row execute function public.iosi_marcar_atualizado();

-- ---------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------
alter table public.snapshots  enable row level security;
alter table public.previsoes  enable row level security;
alter table public.execucoes  enable row level security;

-- ---------------------------------------------------------------------
-- GRANTs explicitos
-- ---------------------------------------------------------------------
grant usage on schema public to anon, authenticated, service_role;

-- comeca do zero para anon/authenticated (rodar de novo nao acumula permissao)
revoke all on public.snapshots, public.previsoes, public.execucoes from anon, authenticated;

-- robo (service_role): leitura e escrita nas tres
grant select, insert, update, delete on public.snapshots, public.previsoes, public.execucoes to service_role;
grant usage, select on sequence public.snapshots_id_seq, public.execucoes_id_seq to service_role;

-- leitura publica so de previsoes
grant select on public.previsoes to anon;

-- a funcao do trigger nao precisa ser chamada pela API
revoke execute on function public.iosi_marcar_atualizado() from public, anon, authenticated;

-- ---------------------------------------------------------------------
-- Policies
-- ---------------------------------------------------------------------
drop policy if exists "previsoes: leitura publica" on public.previsoes;
create policy "previsoes: leitura publica" on public.previsoes
    for select to anon
    using (true);
