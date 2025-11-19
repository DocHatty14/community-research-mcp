@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Community Research MCP - Setup
echo For Claude Desktop
echo ========================================
echo.

cd /d "%~dp0"
set "SERVER_DIR=%CD%"
set "SERVER_FILE=%SERVER_DIR%\community_research_mcp.py"

:: Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
    goto :install
)
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py"
    goto :install
)
echo ERROR: Python not found! Install from python.org
pause
exit /b 1

:install
echo Python found!
echo.
echo [2/4] Installing dependencies...
%PYTHON_CMD% -m pip install -q --upgrade pip
%PYTHON_CMD% -m pip install -q mcp fastmcp httpx pydantic
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed!

:setup_env
echo.
echo [3/4] Setting up API key...
if not exist .env (
    echo Creating .env file...
    (
        echo # Add your Gemini API key here
        echo # Get free key: https://makersuite.google.com/app/apikey
        echo GEMINI_API_KEY=
    ) > .env
    
    echo.
    echo ============================================
    echo ADD YOUR API KEY
    echo ============================================
    echo.
    echo Get FREE Gemini key ^(1,500 requests/day^):
    echo https://makersuite.google.com/app/apikey
    echo.
    echo Opening .env file - add your key after GEMINI_API_KEY=
    echo Then SAVE and CLOSE the file.
    echo.
    pause
    notepad .env
) else (
    echo .env already exists - skipping
)

:configure_claude
echo.
echo [4/4] Configuring Claude Desktop...

set "CLAUDE_CONFIG=%APPDATA%\Claude\claude_desktop_config.json"

:: Create Claude directory if it doesn't exist
if not exist "%APPDATA%\Claude" (
    echo Creating Claude config directory...
    mkdir "%APPDATA%\Claude"
)

:: Check if config exists
if exist "%CLAUDE_CONFIG%" (
    echo Found existing config - backing up...
    copy "%CLAUDE_CONFIG%" "%CLAUDE_CONFIG%.backup" >nul 2>&1
    
    :: Use Python to merge JSON properly
    echo Updating configuration...
    %PYTHON_CMD% -c "import json; import sys; config_path = r'%CLAUDE_CONFIG%'; server_path = r'%SERVER_FILE%'; config = json.load(open(config_path)) if config_path else {}; config.setdefault('mcpServers', {})['community-research'] = {'command': 'python', 'args': [server_path]}; json.dump(config, open(config_path, 'w'), indent=2); print('Config updated!')"
) else (
    echo Creating new config file...
    %PYTHON_CMD% -c "import json; server_path = r'%SERVER_FILE%'; config = {'mcpServers': {'community-research': {'command': 'python', 'args': [server_path]}}}; json.dump(config, open(r'%CLAUDE_CONFIG%', 'w'), indent=2); print('Config created!')"
)

if %errorlevel% neq 0 (
    echo.
    echo WARNING: Auto-config failed. Manual setup required.
    echo.
    echo Add this to: %CLAUDE_CONFIG%
    echo.
    echo {
    echo   "mcpServers": {
    echo     "community-research": {
    echo       "command": "python",
    echo       "args": ["%SERVER_FILE:\=\\%"]
    echo     }
    echo   }
    echo }
    echo.
    pause
    notepad "%CLAUDE_CONFIG%"
) else (
    echo Configuration updated successfully!
)

:complete
echo.
echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo Server location: %SERVER_FILE%
echo Config location: %CLAUDE_CONFIG%
echo.
echo NEXT STEPS:
echo.
echo 1. Make sure you added your API key to .env
echo    ^(Get free key: https://makersuite.google.com/app/apikey^)
echo.
echo 2. CLOSE Claude Desktop if it's running
echo.
echo 3. START Claude Desktop again
echo.
echo 4. Test it with: "use get_server_context"
echo.
echo 5. Then try: "search for Python FastAPI solutions"
echo.
echo.
echo TROUBLESHOOTING:
echo - Server not showing? Check .env has your API key
echo - Still not working? See config at: %CLAUDE_CONFIG%
echo - Need help? Open CLAUDE_DESKTOP_SETUP.md
echo.
pause
