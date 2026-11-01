@echo off
REM 牢图棒 · 基线校准器启动脚本
REM 双击即可，会在浏览器打开 http://127.0.0.1:8848

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 没找到 .venv，请先运行：
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install numpy pillow opencv-python scipy onnxruntime onnx pyyaml jsonschema fastapi uvicorn python-multipart psutil
    pause
    exit /b 1
)

set PYTHONPATH=%~dp0

echo 正在启动校准器...
echo 地址: http://127.0.0.1:8848
echo 关闭此窗口即停止服务。
echo.

start "" http://127.0.0.1:8848
".venv\Scripts\python.exe" -m uvicorn app.server:app --host 127.0.0.1 --port 8848

pause
