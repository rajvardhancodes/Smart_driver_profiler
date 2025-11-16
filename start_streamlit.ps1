Write-Host "Starting Smart Driver Profiler..." -ForegroundColor Green
Write-Host ""
Write-Host "Streamlit will open in your browser automatically." -ForegroundColor Yellow
Write-Host "If it doesn't, go to: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host ""

streamlit run app.py --server.headless false

