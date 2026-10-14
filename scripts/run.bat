@echo off
rem CIFAR10 分类训练一键启动（Windows）
rem 用法: 双击运行，或命令行执行 run.bat [配置文件]
setlocal

if "%1"=="" (
    set CONFIG=confits\default.yaml
) else (
    set CONFIG=%1
)

echo ==========================================
echo  CIFAR10 Classification Training
echo  Config: %CONFIG%
echo ==========================================

python scripts\train.py --config %CONFIG%

if errorlevel 1 (
    echo.
    echo [错误] 训练失败，请检查上方报错信息。
    pause
)
endlocal
