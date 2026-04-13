@echo off
cd /d "%~dp0"
call .\phishing_env\Scripts\activate.bat
python train_optimized.py --mode quick --skip-deep-learning --verbose
pause
