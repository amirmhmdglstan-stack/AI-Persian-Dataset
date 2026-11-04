@echo off
REM ===========================================================================
REM Persian Dataset V2 launcher safety guard.
REM V2 repository: D:\GPT\dataset\repo_v2_review
REM V1 repository: D:\GPT\dataset\repo (MUST NOT be used or branch-switched)
REM ===========================================================================
setlocal
set "PY=D:\GPT\venv\Scripts\python.exe"
set "REPO=D:\GPT\dataset\repo_v2_review"
set "CONFIG=%REPO%\dataset_v2\configs\dataset_config.json"
set "META=%REPO%\dataset_v2\sources\source_metadata.json"
set "TOK=%REPO%\dataset_v2\tokenizer\persian-bpe-32k.json"

echo Repository: %REPO%
echo Python:     %PY%
echo.

REM Refuse the V1 path in either slash style, with or without a trailing slash.
if /I "%REPO%"=="D:\GPT\dataset\repo" goto :v1_repo
if /I "%REPO%"=="D:\GPT\dataset\repo\" goto :v1_repo
if /I "%REPO%"=="D:/GPT/dataset/repo" goto :v1_repo
if /I "%REPO%"=="D:/GPT/dataset/repo/" goto :v1_repo

if not exist "%PY%" goto :no_python
if not exist "%REPO%\dataset_v2" goto :missing_v2
if not exist "%REPO%\dataset_v2\configs\dataset_config.json" goto :missing_v2
cd /d "%REPO%"
if errorlevel 1 goto :fail

REM Verify the repository. This launcher NEVER performs automatic
REM remote updates or branch switching; V1's branch is never touched.
set "CURRENT_BRANCH="
for /f "delims=" %%B in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%B"
if not defined CURRENT_BRANCH goto :not_git

REM Check tracked changes before any Git operation (there are none in this
REM launcher). Generated report files and the smoke workdir are excluded.
set "DIRTY="
for /f "delims=" %%L in ('git status --porcelain --untracked-files=no -- . ":(exclude)dataset_v2/reports" ":(exclude)dataset_v2/.smoke_work" 2^>nul') do set "DIRTY=1"
if defined DIRTY goto :dirty_tree
echo Git guard: branch %CURRENT_BRANCH%; tracked source/config tree clean. Automatic remote updates and branch switching are DISABLED.
echo Update the V2 repository manually before running; V1's branch is never switched.
echo.

REM Core-only install/check. Optional pyarrow/py7zr are NEVER auto-installed.
"%PY%" -m pip install -r "%REPO%\dataset_v2\requirements.txt"
if errorlevel 1 goto :fail
set "WORK=D:\GPT\dataset\v2_work_AB"
set "OUT=D:\GPT\dataset\v2_final_AB"
set "PROJECT=D:\GPT\project"
set "REPORTS=%OUT%\reports"
"%PY%" "%REPO%\dataset_v2\pipeline\check_requirements.py" --config "%CONFIG%" --metadata "%META%" --tokenizer-baseline "%TOK%" --profiles A B
if errorlevel 1 goto :fail

if not exist "%WORK%" mkdir "%WORK%"
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%PROJECT%" mkdir "%PROJECT%"
if not exist "%REPORTS%" mkdir "%REPORTS%"

REM PROFILE B is CC BY-NC-SA / non-commercial and written to a SEPARATE profile_B.
REM This rebuilds A+B from scratch in a separate workdir so global dedup covers both.
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
"%PY%" "%REPO%\dataset_v2\pipeline\compare_tokenizers.py" --config "%CONFIG%" --sample "%OUT%\profile_A\train.txt" --tokenizer-baseline "%TOK%" --reports-dir "%REPORTS%" --out "%REPORTS%\tokenizer_comparison.json" --max-chars 8000000 --scope full_corpus_sample
if errorlevel 1 goto :fail

echo.
echo A+B BUILD COMPLETE. profile_A and profile_B are physically separate.
echo PROFILE B IS NON-COMMERCIAL RESEARCH-ONLY. Do not merge it into profile_A.
goto :eof

:v1_repo
echo ERROR: REPO points to the V1 repository D:\GPT\dataset\repo.
echo Use the dedicated V2 repository D:\GPT\dataset\repo_v2_review instead.
goto :fail

:no_python
echo ERROR: Python executable not found: %PY%
goto :fail

:missing_v2
echo ERROR: dataset_v2 tree or config not found under %REPO%.
echo Clone/check out the V2 repository at D:\GPT\dataset\repo_v2_review first.
goto :fail

:not_git
echo ERROR: %REPO% is not a Git working tree or git is unavailable.
goto :fail

:dirty_tree
echo ERROR: tracked source/config files have local changes.
echo Commit or stash them before running. This launcher will not update or switch over a dirty tree.
goto :fail

:fail
echo.
echo V2 LAUNCHER FAILED. See the command output above. Do not train on incomplete or unverified data.
exit /b 1
