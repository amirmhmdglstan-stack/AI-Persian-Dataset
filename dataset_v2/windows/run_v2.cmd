@echo off
REM ===========================================================================
REM Persian Dataset V2 - FULL PROFILE A build on Windows 11 / Python 3.12
REM Run from a normal CMD window (not Git Bash / PowerShell).
REM All large files are written to D:. V1 root files are never touched.
REM ===========================================================================
setlocal
set "PY=D:\GPT\venv\Scripts\python.exe"
set "REPO=D:\GPT\dataset\repo"
set "WORK=D:\GPT\dataset\v2_work"
set "OUT=D:\GPT\dataset\v2_final"
set "PROJECT=D:\GPT\project"
set "CONFIG=%REPO%\dataset_v2\configs\dataset_config.json"
set "META=%REPO%\dataset_v2\sources\source_metadata.json"
set "TOK=%REPO%\dataset_v2\tokenizer\persian-bpe-32k.json"
set "REPORTS=%OUT%\reports"
set "BRANCH=arena/62203741-ai-persian-dataset"

cd /d "%REPO%"
if errorlevel 1 goto :fail

REM This session's implementation branch is fixed. Do NOT create
REM persian-dataset-v2-pipeline; V1 remains on its own remote branch.
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

REM 1. Metadata-only probes; no corpus file is downloaded here.
"%PY%" "%REPO%\dataset_v2\pipeline\discover_sources.py" --config "%CONFIG%" --workdir "%WORK%" --probe --timeout 20
if errorlevel 1 goto :fail

REM 2. Resumable downloads (add --include-optional to recover the 2021 V1 mirror).
"%PY%" "%REPO%\dataset_v2\pipeline\download_sources.py" --config "%CONFIG%" --metadata "%META%" --workdir "%WORK%" --profiles A --timeout 120
if errorlevel 1 goto :fail

REM 3. Streaming extraction. Re-running skips already-verified sources.
"%PY%" "%REPO%\dataset_v2\pipeline\extract_sources.py" --config "%CONFIG%" --metadata "%META%" --workdir "%WORK%" --profiles A
if errorlevel 1 goto :fail

REM 4. Clean, filter, globally deduplicate, balance, split and write.
REM Use --force instead of --resume to discard an old index and rebuild cleanly.
"%PY%" "%REPO%\dataset_v2\pipeline\build_corpus.py" --config "%CONFIG%" --metadata "%META%" --workdir "%WORK%" --extracted "%WORK%\extracted" --outdir "%OUT%" --reports-dir "%REPORTS%" --profiles A --resume --scope full_run
if errorlevel 1 goto :fail

REM 5. Measured statistics using the unchanged V1 32K tokenizer.
"%PY%" "%REPO%\dataset_v2\pipeline\compute_statistics.py" --config "%CONFIG%" --workdir "%WORK%" --outdir "%OUT%" --reports-dir "%REPORTS%" --tokenizer "%TOK%" --profiles A --scope full_run --out "%REPORTS%\dataset_statistics.json"
if errorlevel 1 goto :fail

REM 6. Independent verification; non-zero exit means DO NOT TRAIN.
"%PY%" "%REPO%\dataset_v2\pipeline\verify_dataset.py" --config "%CONFIG%" --workdir "%WORK%" --outdir "%OUT%" --reports-dir "%REPORTS%" --profiles A --scope full_run --out "%REPORTS%\verification_report_v2.json"
if errorlevel 1 goto :fail

REM 7. 8K/16K/32K comparison trained from scratch on the final Persian text.
"%PY%" "%REPO%\dataset_v2\pipeline\compare_tokenizers.py" --config "%CONFIG%" --sample "%OUT%\profile_A\train.txt" --tokenizer-baseline "%TOK%" --reports-dir "%REPORTS%" --out "%REPORTS%\tokenizer_comparison.json" --max-chars 40000000 --scope full_corpus_sample
if errorlevel 1 goto :fail

REM Copy small reports to the project directory for convenience.
if not exist "%PROJECT%\v2_reports" mkdir "%PROJECT%\v2_reports"
xcopy /E /Y "%REPORTS%\*" "%PROJECT%\v2_reports\" >nul

echo.
echo V2 PROFILE A BUILD COMPLETE.
echo Final files: %OUT%\profile_A
echo Reports:     %REPORTS%
echo Review sample_audit.txt and verification_report_v2.json before training.
goto :eof

:fail
echo.
echo V2 BUILD FAILED. Inspect %WORK%\pipeline.log and the command output above.
echo Do not train on incomplete or unverified data.
exit /b 1
