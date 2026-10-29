@echo off
set PYTHONPATH=.
echo Starting AI Fitness Trainer Dashboard...
.\.venv\Scripts\streamlit.exe run src\app.py
pause
