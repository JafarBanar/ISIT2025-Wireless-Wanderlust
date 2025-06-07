$env:PYTHONPATH = "$PSScriptRoot/src"
$venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
$uvicorn = Join-Path $PSScriptRoot 'venv\Scripts\uvicorn.exe'
& $venvPython -m uvicorn src.deployment.model_server:app --reload --host localhost --port 8000 