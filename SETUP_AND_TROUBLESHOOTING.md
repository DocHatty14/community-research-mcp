# Setup & Troubleshooting Guide

## Quick Setup (2 Minutes)

### Step 1: Run the Installer
1. Double-click `initialize.bat`
2. Wait for dependencies to install
3. Add your API key when the .env file opens
4. Save and close

### Step 2: Restart Claude Desktop
- Close Claude Desktop completely
- Open it again

### Step 3: Test It!
In Claude Desktop, type:
```
use get_server_context
```

You should see available tools and workspace info!

Then try:
```
search for Python FastAPI background task solutions
```

---

## Get FREE API Key

You need a Gemini API key (FREE - 1,500 requests/day):

1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key
4. Paste it in `.env` file after `GEMINI_API_KEY=`

No credit card required!

---

## Manual Setup (If Auto Fails)

### Step 1: Install Dependencies
```bash
pip install mcp fastmcp httpx pydantic
```

### Step 2: Create .env File
Create a file named `.env` in this folder:
```
GEMINI_API_KEY=your_key_here
```

### Step 3: Get Full Path to Server

**Windows (CMD):**
```cmd
cd C:\path\to\community-research-mcp
echo %CD%\community_research_mcp.py
```

**Windows (PowerShell):**
```powershell
cd C:\path\to\community-research-mcp
(Get-Location).Path + "\community_research_mcp.py"
```

Copy the full path!

### Step 4: Edit Claude Desktop Config

**Config file location:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

Usually: `C:\Users\YourName\AppData\Roaming\Claude\claude_desktop_config.json`

**If file exists:**
Add this section to the JSON:
```json
{
  "mcpServers": {
    "community-research": {
      "command": "python",
      "args": ["C:\\FULL\\PATH\\community_research_mcp.py"]
    }
  }
}
```

**If file doesn't exist:**
Create it with this content:
```json
{
  "mcpServers": {
    "community-research": {
      "command": "python",
      "args": ["C:\\FULL\\PATH\\community_research_mcp.py"]
    }
  }
}
```

Replace `C:\\FULL\\PATH\\` with your actual path from Step 3!

**Important:** Use double backslashes `\\` in the path!

### Step 5: Restart Claude Desktop
Close and restart Claude Desktop completely.

---

## Common Issues

### "Server not found"
- Check: `%APPDATA%\Claude\claude_desktop_config.json` exists
- Make sure path in config is FULL path (e.g., `C:\\Users\\...`)
- Use double backslashes `\\` in Windows paths

### "No module named 'mcp'"
- Run: `pip install mcp fastmcp httpx pydantic`

### "No API key configured"
- Check `.env` file has your key: `GEMINI_API_KEY=your_key`
- Make sure there's no space after the `=`

### "Python not found"
Use full Python path in config:
```json
{
  "mcpServers": {
    "community-research": {
      "command": "C:\\Python311\\python.exe",
      "args": ["C:\\full\\path\\to\\community_research_mcp.py"]
    }
  }
}
```

---

## Improving Poor Search Results

### Why Results Are Limited

1. **Limited Free APIs**
   - Stack Overflow API: Limited to tagged questions
   - GitHub API: Only searches issues/discussions
   - Reddit API: Hit or miss for technical content

2. **No Google Search Access**
   - Free APIs don't include Google
   - Most comprehensive answers are in Google results

### Solution: Add Better Search APIs

#### Add Serper (BEST FIX - Recommended)
**Gives you Google Search access!**

1. Go to: https://serper.dev/
2. Sign up (free account)
3. Get API key (2,500 free searches/month)
4. Add to `.env`:
   ```
   SERPER_API_KEY=your_key_here
   ```
5. Restart Claude Desktop

**Result:** Access to Google Search = dramatically better results!

#### Add Perplexity
1. Go to: https://www.perplexity.ai/settings/api
2. Get API key (5 free requests/day)
3. Add to `.env`:
   ```
   PERPLEXITY_API_KEY=your_key_here
   ```

#### Add Brave Search
1. Go to: https://brave.com/search/api/
2. Get API key (2,000 free/month)
3. Add to `.env`:
   ```
   BRAVE_SEARCH_API_KEY=your_key_here
   ```

### Available API Keys

The server supports these APIs (add to `.env`):
```bash
# LLM Providers (need at least one)
GEMINI_API_KEY=AIzaSy...
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
PERPLEXITY_API_KEY=pplx-...

# Search APIs (optional but highly recommended)
SERPER_API_KEY=...          # ⭐ BEST for improving results
BRAVE_SEARCH_API_KEY=...
```

---

## Testing & Verification

### Test 1: Check server is running
```
use get_server_context
```

Expected: Returns workspace info, detected languages, and available tools.

### Test 2: Search community
```
search for Python FastAPI background task solutions
```

Expected: Returns 3-5 recommendations with code, community scores, and evidence.

### Verification Checklist

✅ **Check 1: .env File Exists**
```cmd
dir C:\path\to\community-research-mcp\.env
```

✅ **Check 2: API Keys Are Set**
```cmd
type C:\path\to\community-research-mcp\.env
```

✅ **Check 3: Server Recognizes Keys**
In Claude Desktop:
```
use get_server_context
```

Check `available_providers.configured` field

---

## Free API Key Recommendations

### For Best Results (Recommended):

1. **SERPER_API_KEY** (2,500 free/month)
   - Get it: https://serper.dev/
   - Why: Google Search access = 10x better results

2. **GEMINI_API_KEY** (1,500 free/day)
   - Get it: https://makersuite.google.com/app/apikey
   - Why: Free LLM for synthesis

### For Even Better Results:

3. **PERPLEXITY_API_KEY** (5 free/day)
   - Get it: https://www.perplexity.ai/settings/api
   - Why: Built-in web search + LLM

4. **OPENROUTER_API_KEY** ($5 free credit)
   - Get it: https://openrouter.ai/keys
   - Why: Access to 100+ models

---

## Advanced Usage

### Using a Virtual Environment
```json
{
  "mcpServers": {
    "community-research": {
      "command": "C:\\path\\to\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\community_research_mcp.py"]
    }
  }
}
```

### Checking Logs
Windows logs location:
```
%APPDATA%\Claude\logs\
```

### Multiple API Keys
In `.env`:
```
GEMINI_API_KEY=primary_key
OPENAI_API_KEY=backup_key
```

Server will use first available key.

---

## Still Not Working?

1. ✅ Check config file path is correct
2. ✅ Verify full (absolute) path to .py file
3. ✅ Confirm double backslashes in Windows paths
4. ✅ Make sure .env has API key
5. ✅ Test: `python community_research_mcp.py` runs without errors
6. ✅ Restart Claude Desktop after changes
7. ✅ Check Claude Desktop logs for errors

If still stuck, verify each step above.
