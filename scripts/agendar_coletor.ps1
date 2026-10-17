# agendar_coletor.ps1
# Registra no Agendador de Tarefas do Windows a rotina diaria do IOSI:
#   IOSI - coletor 9h15  -> seg a sex, 09:17 (nome mantido): coletor_snapshots.py manha, depois predictor_9h15.py
#   IOSI - coletor 19h   -> seg a sex, 19:00: coletor_snapshots.py noite, depois validador_19h.py
# As duas acoes de cada tarefa rodam em sequencia (a segunda espera a primeira terminar).
# A segunda roda mesmo se a coleta falhar: o preditor usa o candle de 1h das 9h, e o
# validador grava "sem_candle_real".
#
# Uso (PowerShell, sem precisar de administrador):
#   powershell -ExecutionPolicy Bypass -File scripts\agendar_coletor.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\agendar_coletor.ps1 -Remover
#
# A tarefa roda com o usuario atual e so quando ele esta logado. Se o PC estiver
# desligado ou dormindo no horario, roda assim que possivel (StartWhenAvailable);
# o arquivo guarda "coletado_em" e os candles de 5m antigos continuam no Yahoo, entao
# o preco das 9h15 ainda pode ser recuperado numa coleta atrasada no mesmo dia.

param([switch]$Remover)

$ErrorActionPreference = "Stop"
$Pasta  = Split-Path -Parent $PSScriptRoot          # IOSI-experimentos
$Python = "C:\Users\muril\.venvs\iosi-experimentos\Scripts\python.exe"
$Src    = Join-Path $Pasta "src"

$Tarefas = @(
    @{ Nome = "IOSI - coletor 9h15"; Hora = "09:17"; Modo = "manha"; Depois = "predictor_9h15.py" },
    @{ Nome = "IOSI - coletor 19h";  Hora = "19:00"; Modo = "noite"; Depois = "validador_19h.py" }
)

if ($Remover) {
    foreach ($t in $Tarefas) {
        Unregister-ScheduledTask -TaskName $t.Nome -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Removida: $($t.Nome)"
    }
    return
}

if (-not (Test-Path $Python)) { throw "Python nao encontrado: $Python" }
foreach ($f in "coletor_snapshots.py", "predictor_9h15.py", "validador_19h.py") {
    if (-not (Test-Path (Join-Path $Src $f))) { throw "Script nao encontrado: $f" }
}

$Config = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

foreach ($t in $Tarefas) {
    $Acoes = @(
        (New-ScheduledTaskAction -Execute $Python -WorkingDirectory $Pasta `
            -Argument "`"$(Join-Path $Src 'coletor_snapshots.py')`" $($t.Modo)"),
        (New-ScheduledTaskAction -Execute $Python -WorkingDirectory $Pasta `
            -Argument "`"$(Join-Path $Src $t.Depois)`"")
    )
    $Gatilho = New-ScheduledTaskTrigger -Weekly `
        -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $t.Hora
    Register-ScheduledTask -TaskName $t.Nome -Action $Acoes -Trigger $Gatilho -Settings $Config `
        -Description "IOSI: coleta ($($t.Modo)) + $($t.Depois)" -Force | Out-Null
    Write-Host "Registrada: $($t.Nome) (seg-sex $($t.Hora)): coleta $($t.Modo) + $($t.Depois)"
}

Get-ScheduledTask -TaskName "IOSI - coletor*" |
    Select-Object TaskName, State, @{ n = "Proxima execucao"; e = { ($_ | Get-ScheduledTaskInfo).NextRunTime } } |
    Format-Table -AutoSize
