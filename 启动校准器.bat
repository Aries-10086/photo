@echo off
chcp 65001 >nul
setlocal

REM 牢图棒 · 基线校准器
REM 双击即可。服务意外退出会自动重启；关闭此窗口才真正停止。

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [错误] 没找到 .venv
    echo.
    echo 请先在项目目录执行：
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

set PYTHONPATH=%~dp0
set PORT=8848

REM 如果端口已被占用，说明已经在跑，直接开浏览器
netstat -ano | findstr ":%PORT%" | findstr LISTENING >nul 2>&1
if not errorlevel 1 (
    echo 服务已在运行，直接打开浏览器。
    start "" http://127.0.0.1:%PORT%
    timeout /t 2 >nul
    exit /b 0
)

echo ============================================
echo   牢图棒 · 基线校准器
echo   地址: http://127.0.0.1:%PORT%
echo   
echo   关闭此窗口 = 停止服务
echo   服务意外退出会自动重启
echo ============================================
echo.

start "" http://127.0.0.1:%PORT%

:loop
echo [%time%] 启动服务...
".venv\Scripts\python.exe" -m uvicorn app.server:app --host 127.0.0.1 --port %PORT% --log-level warning
set CODE=%errorlevel%

if %CODE% EQU 0 (
    echo [%time%] 服务正常退出。
    goto done
)

echo.
echo [%time%] 服务异常退出 (code %CODE%)，3 秒后重启...
echo          按 Ctrl+C 可停止重试。
timeout /t 3 >nul
echo.
goto loop

:done
echo.
echo 服务已停止。按任意键关闭窗口。
pause >nul
