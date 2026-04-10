# Hologram VSCode Extension - Quick Start Guide

## Prerequisites

1. **Python Environment**
   ```bash
   cd /home/lenovo/projects/hologram
   source .venv/bin/activate
   ```

2. **Dependencies Installed**
   ```bash
   pip install fastapi uvicorn  # Already installed
   ```

## Step 1: Start Hologram Server

Open a terminal and run:

```bash
cd /home/lenovo/projects/hologram
source .venv/bin/activate
python -m uvicorn hologram.server:app --host 127.0.0.1 --port 8000 --reload
```

You should see:
```
╔═══════════════════════════════════════════════════════════╗
║              🌀 Hologram Server Starting                 ║
╠═══════════════════════════════════════════════════════════╣
║  Host: 127.0.0.1                                         ║
║  Port: 8000                                              ║
║  Memory Dir: /home/lenovo/.hologram_memory               ║
╚═══════════════════════════════════════════════════════════╝

INFO:     Uvicorn running on http://127.0.0.1:8000
```

## Step 2: Test Server (Optional)

In another terminal:

```bash
# Health check
curl http://127.0.0.1:8000/

# Ingest test
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"project":"test","text":"Hello Hologram","origin":"manual"}'

# Query test
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"project":"test","text":"hello","top_k":3}'
```

## Step 3: Load Extension in VSCode

### Option A: Development Mode (Recommended for Testing)

1. Open VSCode
2. File → Open Folder → Select `/home/lenovo/projects/hologram/vscode_extension`
3. Press `F5` (or Run → Start Debugging)
4. A new "Extension Development Host" window will open

### Option B: Package and Install

```bash
cd /home/lenovo/projects/hologram/vscode_extension
npm install -g @vscode/vsce
vsce package
# Install the generated .vsix file
```

## Step 4: Use the Extension

In the Extension Development Host window:

### 4.1 Open a Workspace

Open any project folder (e.g., the hologram project itself).

### 4.2 Check Server Connection

Look at the bottom status bar. You should see:
- "Hologram Server: Connected" (if server is running)
- Or a warning message if server is not running

### 4.3 Open Memory Panel

- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type "Hologram: Show Memory Panel"
- Press Enter

You should see:
- **Total Concepts**: ~27-30 (system concepts auto-loaded)
- **Tier 1 (Domain)**: 0 (no project content yet)
- **Tier 2 (System)**: ~27 (Hologram architecture concepts)

### 4.4 Ingest Content

- Edit and save any file in your workspace
- The extension will automatically ingest it
- Check the server terminal - you'll see the ingestion log
- Refresh the Memory Panel to see updated counts

### 4.5 Query Memory

**Method 1: Command Palette**
- Press `Ctrl+Shift+P`
- Type "Hologram: Query Memory"
- Enter your query (e.g., "concept drift")
- Results appear in the Memory Panel

**Method 2: Memory Panel Search**
- Open Memory Panel
- Type in the search box
- Press Enter
- Results appear below

## Step 5: Verify Tier System

### Check Tier 2 System Concepts

```bash
curl http://127.0.0.1:8000/memory/your_project_name
```

You should see:
```json
{
  "project": "your_project_name",
  "total_concepts": 30+,
  "tier1_count": X,
  "tier2_count": 27,
  "projects": ["your_project_name", "hologram"],
  "recent_traces": [...]
}
```

### Verify Auto-Ingestion

1. Create a new file: `test.py`
2. Add content:
   ```python
   def fibonacci(n):
       return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)
   ```
3. Save the file
4. Check server logs - should see ingestion
5. Query "fibonacci" in Memory Panel

## Troubleshooting

### Server Not Running

**Symptom:** Warning message in VSCode status bar

**Solution:**
```bash
cd /home/lenovo/projects/hologram
source .venv/bin/activate
python -m uvicorn hologram.server:app --host 127.0.0.1 --port 8000 --reload
```

### Extension Not Loading

**Symptom:** Commands not appearing in Command Palette

**Solution:**
1. Check VSCode Developer Tools: Help → Toggle Developer Tools
2. Look for errors in Console
3. Reload window: `Ctrl+R` in Extension Development Host

### TypeScript Compilation Errors

**Symptom:** Extension fails to load

**Solution:**
```bash
cd /home/lenovo/projects/hologram/vscode_extension
npm install
npm run compile
```

Check for errors in the output.

### No Concepts Showing

**Symptom:** Memory Panel shows 0 concepts

**Solution:**
1. Verify server is running
2. Check server logs for errors
3. Try manual ingestion:
   ```bash
   curl -X POST http://127.0.0.1:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"project":"test","text":"test content","origin":"manual"}'
   ```
4. Refresh Memory Panel

## Next Steps

- **Phase 5**: Implement AI integration (Gemini/ChatGPT)
- **Phase 6**: Add project switching UI
- **Phase 7**: Advanced features (CodeLens, drift viz, Git integration)

## Testing Checklist

- [ ] Server starts without errors
- [ ] Extension loads in Development Host
- [ ] Status bar shows "Connected"
- [ ] Memory Panel opens
- [ ] Tier 2 concepts auto-loaded (~27 concepts)
- [ ] File save triggers auto-ingestion
- [ ] Query returns results
- [ ] Memory Panel updates on refresh

## Support

For issues, check:
1. Server logs (terminal running uvicorn)
2. VSCode Developer Tools (Help → Toggle Developer Tools)
3. Extension Host logs (in Development Host window)
