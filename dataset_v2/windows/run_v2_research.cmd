@echo off
REM ===========================================================================
REM Persian Dataset V2 - PROFILE A + PROFILE B build (research-only extension)
REM PROFILE B is CC BY-NC-SA / non-commercial. It is written to a SEPARATE
REM profile_B directory and must NEVER be merged into profile_A.
REM This rebuilds A+B from scratch in a separate workdir so global deduplication
REM covers both profiles. PROFILE C is never downloaded or trained on.
REM ===========================================================================
setlocal
set "PY=D:\GPT\venv\Scripts\python.exe"
set "REPO=D:\GPT\dataset\repo"
set "WORK=D:\GPT\dataset\v2_work_AB"
set "OUT=D:\GPT\dataset\v2_final_AB"
set "PROJECT=D:\GPT\project"
set "CONFIG=%REPO%\dataset_v2\configs\dataset_config.json"
set "META=%REPO%\dataset_v2\sources\source_metadata.json"
set "TOK=%REPO%\dataset_v2\tokenizer\persian-bpe-32k.json"
set "REPORTS=%OUT%\reports"
set "BRANCH=arena/62203741-ai-persian-dataset"

cd /d "%REPO%"
if errorlevel 1 goto :fail
git fetch origin %BRANCH%
if errorlevel 1 goto :fail
git checkout %BRANCH%
if errorlevel 1 goto :fail
git pull --ff-only origin %BRANCH%
if errorlevel 1 goto :fail
"%PY%" -m pip install -r "%REPO%\dataset_v2\requirements.txt"
if errorlevel 1 goto :fail
if not exist "%WORK%" mkdir "%WORK%"
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%PROJECT%" mkdir "%PROJECT%"
if not exist "%REPORTS%" mkdir "%REPORTS%"

"%PY%" "%REPO%\dataset_v2\pipeline\discover_sources.py" --config "%CONFIG%" --workdir "%WORK%" --probe --timeout 20
if errorlevel 1 goto :fail
"%PY%" "%REPO%\dataset_v2\pipeline\download_sources.py" --config "%CONFIG%" --metadata "%META%" --workdir "%WORK%" --profiles A B --timeout 120
if errorlevel 1 goto :fail
"%PY%" "%REPO%\dataset_v2\pipeline\extract_sources.py" --config "%CONFIG%" --metadata "%META%" --workdir "%WORK%" --profiles A B
if errorlevel 1 goto :fail
"%PY%" "%REPO%\dataset_v2\pipeline\build_corpus.py" --config "%CONFIG%" --metadata "%META%" --workdir "%WORK%" --extracted "%WORK%\extracted" --outdir "%OUT%" --reports-dir "%REPORTS%" --profiles A B --force --scope full_run
if errorlevel 1 goto :fail
"%PY%" "%REPO%\dataset_v2\pipeline\compute_statistics.py" --config "%CONFIG%" --workdir "%WORK%" --outdir "%OUT%" --reports-dir "%REPORTS%" --tokenizer "%TOK%" --profiles A B --scope full_run --out "%REPORTS%\dataset_statistics.json"
if errorlevel 1 goto :fail
"%PY%" "%REPO%\dataset_v2\pipeline\verify_dataset.py" --config "%CONFIG%" --workdir "%WORK%" --outdir "%OUT%" --reports-dir "%REPORTS%" --profiles A B --scope full_run --out "%REPORTS%\verification_report_v2.json"
if errorlevel 1 goto :fail
"%PY%" "%REPO%\dataset_v2\pipeline\compare_tokenizers.py" --config "%CONFIG%" --sample "%OUT%\profile_A\train.txt" --tokenizer-baseline "%TOK%" --reports-dir "%REPORTS%" --out "%REPORTS%\tokenizer_comparison.json" --max-chars 40000000 --scope full_corpus_sample
if errorlevel 1 goto :fail

echo.
echo A+B BUILD COMPLETE. profile_A and profile_B are physically separate.
echo PROFILE B IS NON-COMMERCIAL RESEARCH-ONLY. Do not merge it into profile_A.
goto :eof

:fail
echo A+B BUILD FAILED. Inspect %WORK%\pipeline.log.
exit /b 1
