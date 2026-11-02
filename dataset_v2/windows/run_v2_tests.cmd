@echo off
REM ===========================================================================
REM Persian Dataset V2 launcher safety guard.
REM V2 repository: D:\GPT\dataset\repo_v2_review
REM V1 repository: D:\GPT\dataset\repo (MUST NOT be used or branch-switched)
REM ===========================================================================
setlocal
set "PY=D:\GPT\venv\Scripts\python.exe"
set "REPO=D:\GPT\dataset\repo_v2_review"
set "BRANCH=arena/62203741-ai-persian-dataset"
set "CONFIG=%REPO%\dataset_v2\configs\dataset_config.json"
set "META=%REPO%\dataset_v2\sources\source_metadata.json"
set "TOK=%REPO%\dataset_v2\tokenizer\persian-bpe-32k.json"

echo Repository: %REPO%
echo Branch:     %BRANCH%
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

REM Verify the fixed session branch. This launcher NEVER performs automatic
REM remote updates or branch switching; V1's branch is never touched.
set "CURRENT_BRANCH="
for /f "delims=" %%B in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%B"
if not defined CURRENT_BRANCH goto :not_git
if /I not "%CURRENT_BRANCH%"=="%BRANCH%" goto :wrong_branch

REM Check tracked changes before any Git operation (there are none in this
REM launcher). Generated report files and the smoke workdir are excluded.
set "DIRTY="
for /f "delims=" %%L in ('git status --porcelain --untracked-files=no -- . ":(exclude)dataset_v2/reports" ":(exclude)dataset_v2/.smoke_work" 2^>nul') do set "DIRTY=1"
if defined DIRTY goto :dirty_tree
echo Git guard: branch OK; tracked source/config tree clean. Automatic remote updates and branch switching are DISABLED.
echo Update the V2 repository manually before running; V1's branch is never switched.
echo.

REM Core-only install/check. Optional pyarrow/py7zr are NEVER auto-installed.
"%PY%" -m pip install -r "%REPO%\dataset_v2\requirements.txt"
if errorlevel 1 goto :fail
set "WORK=D:\GPT\dataset\v2_work\smoke"
set "SAMPLE=%REPO%\dataset_v2\tests\fixtures\representative_persian_sample.txt"
"%PY%" "%REPO%\dataset_v2\pipeline\check_requirements.py" --config "%CONFIG%" --metadata "%META%" --tokenizer-baseline "%TOK%" --profiles A
if errorlevel 1 goto :fail

REM Unit tests and small-sample integration test only. No corpus download.
REM Reports under dataset_v2\reports are regenerated and labelled small_sample.
"%PY%" "%REPO%\dataset_v2\tests\run_smoke_test.py" --python "%PY%" --workdir "%WORK%" --tokenizer-sample "%SAMPLE%" --tokenizer-max-chars 20000
if errorlevel 1 goto :fail
echo.
echo V2 UNIT + SMALL-SAMPLE INTEGRATION TESTS PASSED.
echo See %REPO%\dataset_v2\reports\unit_test_output.txt
echo See %REPO%\dataset_v2\reports\small_sample_statistics.json
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

:wrong_branch
echo ERROR: current branch is "%CURRENT_BRANCH%", expected "%BRANCH%".
echo This launcher will not switch branches. Check out the correct V2 branch manually.
goto :fail

:dirty_tree
echo ERROR: tracked source/config files have local changes.
echo Commit or stash them before running. This launcher will not update or switch over a dirty tree.
goto :fail

:fail
echo.
echo V2 LAUNCHER FAILED. See the command output above. Do not train on incomplete or unverified data.
exit /b 1
