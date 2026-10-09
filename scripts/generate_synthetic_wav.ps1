$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Output = Join-Path $Root "data\synthetic_russian_tts.wav"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null
Add-Type -AssemblyName System.Speech
$Synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$Russian = $Synth.GetInstalledVoices() | Where-Object {
    $_.VoiceInfo.Culture.Name -like "ru-*"
} | Select-Object -First 1
if ($Russian) {
    $Synth.SelectVoice($Russian.VoiceInfo.Name)
}
$Synth.Rate = -1
$Synth.SetOutputToWaveFile($Output)
$TextBytes = [Convert]::FromBase64String("0J/RgNC40LLQtdGCLiDQrdGC0L4g0LHQtdC30L7Qv9Cw0YHQvdCw0Y8g0YHQuNC90YLQtdGC0LjRh9C10YHQutCw0Y8g0LfQsNC/0LjRgdGMINC00LvRjyDQv9GA0L7QstC10YDQutC4INGB0LXRgNCy0LjRgdCwINGA0LDRgdC/0L7Qt9C90LDQstCw0L3QuNGPINCz0L7Qu9C+0YHQsC4g0JzRiyDQv9GA0L7QstC10YDRj9C10Lwg0YDQtdCz0LjRgdGC0YDQsNGG0LjRjiwg0L/QvtC40YHQuiDQs9C+0LLQvtGA0Y/RidC10LPQviDQuCDRgNCw0YHRiNC40YTRgNC+0LLQutGDINGA0YPRgdGB0LrQvtC5INGA0LXRh9C4LiDQrdGC0LAg0LfQsNC/0LjRgdGMINC90LUg0L/QvtC00YLQstC10YDQttC00LDQtdGCINC60LDRh9C10YHRgtCy0L4g0YDQsNCx0L7RgtGLINC90LAg0YDQtdCw0LvRjNC90L7QvCDQs9C+0LvQvtGB0LUg0YfQtdC70L7QstC10LrQsC4=")
$Text = [Text.Encoding]::UTF8.GetString($TextBytes)
$Synth.Speak($Text)
$Synth.Dispose()
Write-Host $Output

