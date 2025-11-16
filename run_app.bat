@echo off
echo ========================================
echo  Smart Driver Profiler Application
echo ========================================
echo.
echo Starting Streamlit server...
echo.
echo The app will open in your browser automatically.
echo If not, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

cd /d "%~dp0"
streamlit run app.py

pause


