@echo off
REM ===========================================================================
REM Persian Dataset V2 - unit tests and small-sample integration test.
REM No external corpus is downloaded. Output reports in the repository under
REM dataset_v2\reports are regenerated and labelled small_sample.
REM ===========================================================================
setlocal
set "PY=D:\GPT\venv\Scripts\python.exe"
set "REPO=D:\GPT\dataset\repo"
set "WORK=D:\GPT\dataset\v2_work\smoke"
set "SAMPLE=%REPO%\dataset_v2\tests\fixtures\representative_persian_sample.txt"

cd /d "%REPO%"
if errorlevel 1 goto :fail
"%PY%" -m pip install -r "%REPO%\dataset_v2\requirements.txt"
if errorlevel 1 goto :fail
"%PY%" "%REPO%\dataset_v2\tests\run_smoke_test.py" --python "%PY%" --workdir "%WORK%" --tokenizer-sample "%SAMPLE%" --tokenizer-max-chars 20000
if errorlevel 1 goto :fail
echo.
echo V2 UNIT + SMALL-SAMPLE INTEGRATION TESTS PASSED.
echo See %REPO%\dataset_v2\reports\unit_test_output.txt
echo See %REPO%\dataset_v2\reports\small_sample_statistics.json
goto :eof

:fail
echo V2 TESTS FAILED. Inspect %WORK%\pipeline.log and integration_test_output.txt.
exit /b 1
