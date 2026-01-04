# Troubleshooting Guide

## PyTorch DLL Error (Windows Error 1114)

If you see the warning: **"LLM Features Unavailable: langchain-openai is not available (possibly due to PyTorch DLL error)"**, the application is running in **template-based mode**. This means:

✅ **What Still Works:**
- Drug interaction checking (graph-based detection)
- Interaction risk level assessment
- Template-based explanations
- All visualizations and statistics
- Core functionality

❌ **What's Disabled:**
- AI-generated explanations (LLM features)
- Enhanced RAG-powered responses

### Solution 1: Install Visual C++ Redistributables (Recommended)

1. Download Visual C++ Redistributables:
   - **Direct link**: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Or search for "Visual C++ Redistributables 2022" on Microsoft's website

2. Run the installer and restart your computer

3. Try running the application again:
   ```bash
   streamlit run streamlit_app.py
   ```

### Solution 2: Reinstall PyTorch

If Solution 1 doesn't work, try reinstalling PyTorch:

```bash
# Uninstall PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Or if you have CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Solution 3: Use Template Mode (No Fix Needed)

The application works perfectly fine in template mode! You can:
- Check drug interactions
- View risk levels and severity
- See detailed interaction information
- Use all visualization features

The only difference is that explanations are template-based rather than AI-generated, which is often sufficient for most use cases.

### Verification

After applying a fix, restart the application. If the warning disappears, AI features are enabled. If it still appears, the application will continue working in template mode.

## Invalid API Key Error (401)

If you see an error like: **"Incorrect API key provided"** or **"Error code: 401"**:

### What This Means
- Your API key is invalid, expired, or incorrect
- The application will automatically switch to **template-based mode**
- Core functionality (drug interaction checking) still works perfectly

### Solutions

#### Option 1: Get a Valid API Key

**For OpenRouter:**
1. Go to https://openrouter.ai/keys
2. Sign up or log in
3. Create a new API key
4. Set it as an environment variable:
   ```bash
   # Windows PowerShell
   $env:OPENROUTER_API_KEY="your_key_here"
   
   # Windows CMD
   set OPENROUTER_API_KEY=your_key_here
   
   # Linux/Mac
   export OPENROUTER_API_KEY="your_key_here"
   ```

**For OpenAI:**
1. Go to https://platform.openai.com/account/api-keys
2. Sign up or log in
3. Create a new API key
4. Set it as an environment variable:
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="your_key_here"
   
   # Windows CMD
   set OPENAI_API_KEY=your_key_here
   
   # Linux/Mac
   export OPENAI_API_KEY="your_key_here"
   ```

#### Option 2: Use .env File
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_key_here
```
or
```
OPENAI_API_KEY=your_key_here
```

#### Option 3: Continue Without API Key
The application works perfectly in template mode without an API key. You can:
- Check drug interactions
- View risk levels and severity
- See template-based explanations
- Use all visualization features

### Verification
After setting a new API key, restart the Streamlit app. The warnings should disappear if the key is valid.

## Other Common Issues

### API Key Not Found

If you see an error about API keys:
- The application can work without an API key in template mode
- To enable AI features, set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` environment variable
- Create a `.env` file in the project root with:
  ```
  OPENROUTER_API_KEY=your_key_here
  ```

### Dataset Not Found

If the application can't find the dataset:
- Place your DDInter dataset in the `data/` directory
- Or the application will use sample data if available
- Check that the file path in `streamlit_app.py` matches your dataset location

### Import Errors

If you see import errors:
```bash
pip install -r requirements.txt
```

Make sure all dependencies are installed.

## Getting Help

If you continue to experience issues:
1. Check the error message in the terminal/console
2. Verify all dependencies are installed: `pip list`
3. Try running in a fresh virtual environment
4. Check Python version (3.8+ recommended)


