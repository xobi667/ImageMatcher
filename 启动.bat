@echo off
chcp 65001 >nul 2>&1
title ImageMatcher v2.0
cd /d "%~dp0"

:: 检查依赖
py -3 -c "import flask, PIL, numpy, scipy" >nul 2>&1
if %errorlevel% neq 0 (
    echo 首次运行，安装依赖中...
    py -3 -m ensurepip --upgrade >nul 2>&1
    py -3 -m pip install flask pillow numpy scipy -i https://pypi.tuna.tsinghua.edu.cn/simple -q
    echo 安装完成！
)

py -3 app.py
pause
