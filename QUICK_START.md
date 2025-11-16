# Quick Start Guide

## How to Run the Application

### Method 1: Double-click the batch file (Easiest)
1. Double-click `run_app.bat` in Windows Explorer
2. Wait for Streamlit to start (it will open your browser automatically)
3. The app will be available at: **http://localhost:8501**

### Method 2: Run from Terminal/Command Prompt
1. Open Command Prompt or PowerShell in this folder
2. Run the command:
   ```
   streamlit run app.py
   ```
3. The terminal will show a URL (usually http://localhost:8501)
4. Open that URL in your browser

### Method 3: Run from PowerShell
1. Open PowerShell in this folder
2. Run:
   ```
   .\start_streamlit.ps1
   ```
   Or simply:
   ```
   streamlit run app.py
   ```

## Troubleshooting

### If you see "localhost refused to connect":
1. **Make sure Streamlit is actually running** - Check the terminal/command prompt window
2. **Wait 10-15 seconds** after starting - Streamlit needs time to initialize
3. **Check the URL** - Make sure you're going to `http://localhost:8501` (not https)
4. **Check if port 8501 is already in use** - Another application might be using it
5. **Try a different port**:
   ```
   streamlit run app.py --server.port 8502
   ```
   Then go to: http://localhost:8502

### If you see errors about missing files:
1. Make sure `driver_data.csv` exists - Run `python generate_data.py` first if needed

### If the browser doesn't open automatically:
1. Manually open your browser
2. Go to: http://localhost:8501
3. Or check the terminal for the exact URL shown

## What to Expect

When Streamlit starts successfully, you should see:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

## Stopping the Server

- Press `Ctrl+C` in the terminal/command prompt window
- Or close the terminal window

## Need Help?

If you're still having issues:
1. Make sure all packages are installed: `pip install -r requirements.txt`
2. Make sure the data file exists: `python generate_data.py`
3. Check that Python and Streamlit are installed correctly
4. Try running from a different terminal/command prompt

