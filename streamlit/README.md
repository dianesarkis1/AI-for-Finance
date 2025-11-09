# Investment Memo Generator - Streamlit Interface

A comprehensive web interface for generating investment memos from various document types using Claude Sonnet 4.

## Features

- **Multiple Input Types**:
  - 📄 PDF files
  - 📝 Text files (.txt, .md)
  - 📋 JSON/JSONL files (automatically extracts credit agreements)
  - 🌐 URLs (fetches web content)

- **Flexible Prompt System**:
  - Choose from pre-existing prompt templates in `/prompts`
  - Or create custom prompts on-the-fly

- **Claude Sonnet 4 Integration**:
  - Uses the latest Claude Sonnet 4 model (`claude-sonnet-4-20250514`)
  - Supports up to 16,000 output tokens

- **User-Friendly Interface**:
  - Clean, intuitive design
  - Document preview
  - Download generated memos as Markdown or Text

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your Anthropic API key**:

   You can either:
   - Set the `ANTHROPIC_API_KEY` environment variable:
     ```bash
     export ANTHROPIC_API_KEY='your-api-key-here'
     ```
   - Or enter it directly in the Streamlit interface

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the interface**:
   - The app will open automatically in your browser
   - Or navigate to `http://localhost:8501`

3. **Generate a memo**:
   - Choose input method (upload file or enter URL)
   - Provide your document
   - Select or enter a prompt
   - Click "Generate Investment Memo"
   - Download your generated memo

## How It Works

The app leverages existing functions from the main repository:

### Document Processing
- **Text files** (`read_text_file`): Directly reads .txt and .md files
- **JSONL files** (`extract_credit_agreement_from_jsonl`): Extracts and cleans credit agreement text
- **JSON files**: Parses JSON and extracts 'text' field if available
- **PDF files**: Uses PyPDF2 or PyMuPDF to extract text
- **URLs**: Uses requests library to fetch content

### Memo Generation
- **API Communication**:
  - `build_anthropic_payload`: Constructs API request
  - `call_anthropic_api`: Sends request to Anthropic API
  - `extract_output_text_anthropic`: Extracts text from response

### Prompt Templates
- Loads all `.txt` files from `/prompts` directory
- Allows custom prompts for flexibility

## Configuration

### Model Settings
- **Model**: `claude-sonnet-4-20250514` (Claude Sonnet 4)
- **Max Output Tokens**: 16,000
- **Temperature**: Default (controlled by API)

### File Support
The app validates and processes these file types:
- `.pdf` - Requires PyPDF2 or PyMuPDF
- `.txt`, `.md` - Plain text formats
- `.json` - JSON format (extracts 'text' field if present)
- `.jsonl` - JSON Lines format (uses credit agreement extraction)

## Project Structure

```
streamlit/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Dependencies

### Required
- `streamlit>=1.28.0` - Web interface framework

### Optional (for specific features)
- `PyPDF2>=3.0.0` OR `PyMuPDF>=1.23.0` - PDF processing
- `requests>=2.31.0` - URL fetching
- `python-dotenv>=1.0.0` - Environment variable management

## Troubleshooting

### "No PDF processing library available"
Install either PyPDF2 or PyMuPDF:
```bash
pip install PyPDF2
# or
pip install PyMuPDF
```

### "requests library not installed"
Install requests:
```bash
pip install requests
```

### "API Key not found"
Make sure you've either:
1. Set the `ANTHROPIC_API_KEY` environment variable
2. Entered your API key in the sidebar

### "Failed to extract output"
- Check that your API key is valid
- Ensure you have sufficient API credits
- Verify your internet connection

## Examples

### Example 1: Upload a PDF
1. Select "Upload File"
2. Choose a PDF file
3. Select a prompt template (e.g., "baseline")
4. Click "Generate Investment Memo"

### Example 2: Fetch from URL
1. Select "Enter URL"
2. Enter a URL (e.g., a financial document URL)
3. Click "Fetch URL"
4. Enter a custom prompt
5. Click "Generate Investment Memo"

### Example 3: Process JSONL Credit Agreement
1. Select "Upload File"
2. Upload a .jsonl file from `/data`
3. Select "prompt_gen_anthropic_context" template
4. Click "Generate Investment Memo"

## API Usage

Each memo generation makes one API call to Claude Sonnet 4:
- **Input**: Combined prompt + document text
- **Output**: Up to 16,000 tokens
- **Cost**: Check Anthropic's pricing page for current rates

## Future Enhancements

Potential features for future versions:
- Batch processing multiple documents
- Save/load memo templates
- Export to additional formats (PDF, DOCX)
- Integration with evaluation metrics
- Comparison between different prompts
- History of generated memos

## Support

For issues or questions:
1. Check this README
2. Review the main repository documentation
3. Check Anthropic's API documentation

## License

This interface is part of the AI-for-Finance project.
