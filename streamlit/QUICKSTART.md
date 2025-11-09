# Quick Start Guide

Get up and running with the Investment Memo Generator in 3 simple steps:

## Step 1: Install Dependencies

```bash
cd streamlit
pip install -r requirements.txt
```

## Step 2: Set Your API Key

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Or create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-api-key-here
```

## Step 3: Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## First Memo

Try this workflow for your first memo:

1. **Upload a file**:
   - Click "Browse files"
   - Select a document (PDF, TXT, JSON, or JSONL)

2. **Choose a prompt**:
   - Select a prompt template from the dropdown
   - Or enter "Custom" and write your own

3. **Generate**:
   - Click "🚀 Generate Investment Memo"
   - Wait a few seconds for Claude to process

4. **Download**:
   - Review the generated memo
   - Click "⬇️ Download as Markdown" to save

That's it! You now have a professional investment memo.

## Testing with Sample Data

If you have data in the `/data` directory:

```bash
# From the streamlit folder
streamlit run app.py
```

Then upload a sample file from `../data/` to test the system.

## Troubleshooting

**Can't find dependencies?**
Make sure you're in the `streamlit` directory when running pip install.

**API key not working?**
Double-check your key at https://console.anthropic.com/

**PDF upload not working?**
Run: `pip install PyPDF2` or `pip install PyMuPDF`

**URL fetching not working?**
Run: `pip install requests`

## Next Steps

- Explore different prompt templates in the `/prompts` directory
- Try different document types (PDF vs TXT vs JSONL)
- Experiment with custom prompts
- Compare outputs from different prompts

Need more help? Check the full [README.md](README.md)
