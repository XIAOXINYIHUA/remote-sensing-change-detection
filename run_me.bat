@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ============================================
echo   遥感解译助手 v3
echo ============================================
echo.

if "%1"=="batch" (
    echo [批量处理模式]
    python batch/batch_runner.py %2 %3 %4
) else if "%1"=="scheduler" (
    echo [定时任务模式]
    python batch/scheduler.py
) else if "%1"=="docker" (
    echo [Docker 模式]
    docker-compose -f docker/docker-compose.yml up -d
) else if "%1"=="install" (
    echo [安装依赖]
    pip install -r requirements.txt
    echo Done.
) else if "%1"=="web" (
    echo [Web 界面模式]
    streamlit run app.py --server.port=8501
) else (
    echo [命令行模式]
    python main.py
)

echo.
echo Done.
pause
