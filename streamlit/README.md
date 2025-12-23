# Investment Memo Generator - Streamlit App
A web interface for generating and evaluating investment memos from credit agreements using multiple AI models.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# Optional (for specific features):
pip install PyPDF2  # or PyMuPDF for PDF support
pip install requests  # for URL fetching
```

### 2. Set API Keys
```bash
export ANTHROPIC_API_KEY='your-key-here'
export OPENAI_API_KEY='your-key-here'
export GEMINI_API_KEY='your-key-here'
```

Or create a `.env` file in the project root

### 3. Run the App
```bash
streamlit run streamlit/app.py
```
The app opens at `http://localhost:8501`

---

## Features
### Input Types
- 📄 **PDF files** - Extracts text automatically
- 📝 **Text files** (.txt, .md) - Direct processing
- 📋 **JSON/JSONL files** - Extracts credit agreements
- 🌐 **URLs** - Fetches and processes web content

### AI Models
- **Claude Sonnet 4**
- **GPT-5**
- **Gemini 2.5 Pro**

### Prompts
- Choose from templates in `/prompts` directory
- Or write custom prompts in the interface
- Optional few-shot examples for improved quality

### Evaluation
Real-time evaluation with 4 metrics: accuracy, completeness, consistency, quality

### Output
- View generated memos in-app
- Download as Markdown or Text
- View detailed evaluation results
- Export evaluation JSON

---

### Model Settings
- **Default Model**: Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- **Max Output Tokens**: 16,000
- **Temperature**: Model defaults
---

## Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| Generate memo | 2-5 sec | Typical credit agreement |
| Full evaluation | 2-4 min | 3 evaluators × 4 metrics |
| PDF extraction | <1 sec | Depends on file size |
| URL fetching | 1-3 sec | Depends on network |

## Advanced Features

### Few-Shot Examples
- Enable via checkbox in the interface
- Automatically includes example memos in prompt
- Improves memo quality and consistency

### Custom Prompts
- Write prompts directly in the app's text area
- Or add .txt files to `../prompts/` for reuse
- Templates appear automatically in dropdown

---

## App Architecture

The Streamlit app (`app.py`) provides a web interface that:
1. Handles file uploads and URL fetching
2. Calls memo generation via subprocess to `evals/model_run.py`
3. Runs evaluation via `evals/metrics.py`
4. Displays results and provides download options

All heavy computation happens in the backend evaluation modules.