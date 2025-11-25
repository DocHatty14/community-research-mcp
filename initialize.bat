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
%PYTHON_CMD% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed!

:setup_env
echo.
echo [3/4] Setting up environment...
if not exist .env (
    echo Creating .env file...
    (
        echo # Optional: Enhanced Search APIs
        echo # These are optional - the server works without them
        echo.
        echo # Reddit API ^(for authenticated access^)
        echo REDDIT_CLIENT_ID=
        echo REDDIT_CLIENT_SECRET=
        echo REDDIT_REFRESH_TOKEN=
        echo.
        echo # Premium search APIs ^(optional^)
        echo FIRECRAWL_API_KEY=
        echo TAVILY_API_KEY=
    ) > .env
    echo Created .env file with optional API settings
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
echo 1. CLOSE Claude Desktop if it's running
echo.
echo 2. START Claude Desktop again
echo.
echo 3. Test it with: "use get_server_context"
echo.
echo 4. Then try: "search for Python FastAPI solutions"
echo.
echo.
echo OPTIONAL: Add API keys to .env for enhanced search:
echo - Reddit API for authenticated access
echo - Firecrawl/Tavily for premium web search
echo.
echo TROUBLESHOOTING:
echo - Server not showing? See config at: %CLAUDE_CONFIG%
echo - Need help? Open CLAUDE_DESKTOP_SETUP.md
echo.
pause
