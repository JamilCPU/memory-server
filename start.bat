@echo off
cd /d %~dp0
echo Starting Memory Server on port 8765...
python server.py
pause
