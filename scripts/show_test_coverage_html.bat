
chdir "%~dp0.."

rem remove old htmlcov
@RD /S /Q htmlcov

rem generate html

coverage html

rem host html

cd htmlcov

REM python -m http.server 8888

start python -m http.server 8888

start "" "http://localhost:8888"