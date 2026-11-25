@echo off
title Brain Tumor Classifier - Streamlit Server
cd /d "C:\Users\ponug\.gemini\antigravity\scratch\brain_tumor_classifier"
echo Starting Brain Tumor Classification App...
echo App will open at: http://127.0.0.1:8501
echo.
echo DO NOT CLOSE THIS WINDOW - it keeps the server running.
echo.
.venv\Scripts\streamlit.exe run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
pause
