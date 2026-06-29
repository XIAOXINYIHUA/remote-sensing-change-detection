@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d E:\解译工具
python -u main.py > run_out.log 2>&1
echo Done >> run_out.log
