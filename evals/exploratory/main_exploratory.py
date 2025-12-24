#!/usr/bin/env python3
"""
Main Exploratory Script

This script runs model_run.py on a random subset of 3 files from train.jsonl
with multiple AI models to generate investment memos for exploratory analysis.

PURPOSE:
- Compare investment memo quality across different AI models
- Test model performance on real credit agreement data
- Generate reproducible results for analysis

MODELS TESTED:
- GPT-5 (OpenAI) - Latest GPT model
- Claude Sonnet 4 (Anthropic) - High-quality reasoning
- Gemini 2.5 Pro (Google) - Advanced reasoning capabilities

WORKFLOW:
1. Loads train.jsonl (484 credit agreement records)
2. Randomly selects 3 records (seed=42 for reproducibility)
3. Runs each available model on each record
4. Saves investment memos to data/exploratory_outputs/
5. Generates descriptive filenames: record_XX_[model]_memo.md

REQUIREMENTS:
- At least one API key must be set (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY)
- model_run.py must be in the same directory
- train.jsonl must exist in ../data/

USAGE:
    python main_exploratory.py

OUTPUT:
    data/exploratory_outputs/
    ├── record_01_[source_url].jsonl
    ├── record_02_[source_url].jsonl  
    ├── record_03_[source_url].jsonl
    ├── record_01_gpt_5_memo.md
    ├── record_01_claude_3_sonnet_memo.md
    ├── record_01_gemini_2_5_pro_memo.md
    └── ... (12 total memo files)

Each memo contains:
- Executive Summary (date, company, deal overview, background, purpose)
- Investment Highlights & Risks (key points from investor perspective)  
- Key Deal Information (table with deal size, price, interest rate, covenants, maturity, payment frequency)
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    print(f"🎲 Random seed set to {seed} for reproducibility")


def load_train_data(train_path: Path) -> List[Dict[str, Any]]:
    """Load all records from train.jsonl file."""
    print(f"📂 Loading data from {train_path}...")
    
    records = []
    with train_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
    
    print(f"✅ Loaded {len(records)} records from train.jsonl")
    return records


def select_random_subset(records: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    """Select a random subset of n records."""
    if len(records) < n:
        print(f"⚠️  Warning: Only {len(records)} records available, using all of them")
        return records
    
    selected = random.sample(records, n)
    print(f"🎯 Selected {len(selected)} random records for analysis")
    
    # Print selected records info
    for i, record in enumerate(selected, 1):
        source_url = record.get('source_url', 'Unknown')
        text_length = len(record.get('text', ''))
        print(f"  {i}. {source_url} ({text_length:,} chars)")
    
    return selected


def create_output_directory() -> Path:
    """Create the exploratory outputs directory."""
    output_dir = Path("../data/exploratory_outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    return output_dir


def save_selected_records(selected_records: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    """Save selected records as individual JSONL files for model_run.py."""
    saved_files = []
    
    for i, record in enumerate(selected_records, 1):
        # Create filename from source URL or use index
        source_url = record.get('source_url', f'unknown_{i}')
        # Clean filename by removing invalid characters
        clean_name = "".join(c for c in source_url if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_')[:50]  # Limit length
        
        filename = f"record_{i:02d}_{clean_name}.jsonl"
        file_path = output_dir / filename
        
        # Save as single-record JSONL file
        with file_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        
        saved_files.append(file_path)
        print(f"💾 Saved record {i} to {filename}")
    
    return saved_files


def run_model_on_file(model: str, file_path: Path, output_dir: Path, record_index: int) -> Path:
    """Run model_run.py on a specific file with a specific model."""
    
    # Create output filename
    model_name = model.replace("-", "_").replace(".", "_")
    filename = f"record_{record_index:02d}_{model_name}_memo.md"
    output_path = output_dir / filename
    
    # Build command
    cmd = [
        "python", "model_run.py",
        "--model", model,
        "--input-file", str(file_path),
        "--output", str(output_path)
    ]
    
    print(f"🤖 Running {model} on record {record_index}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Success: {model} completed for record {record_index}")
            print(f"   Output saved to: {filename}")
            return output_path
        else:
            print(f"❌ Error running {model} on record {record_index}:")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {model} took too long on record {record_index}")
        return None
    except Exception as e:
        print(f"💥 Exception running {model} on record {record_index}: {e}")
        return None


def main():
    """Main function to run exploratory analysis."""
    print("=" * 80)
    print("🔬 MAIN EXPLORATORY SCRIPT")
    print("=" * 80)
    
    # Check for API keys
    print("🔑 Checking API keys...")
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"), 
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    }
    
    available_models = []
    for key, value in api_keys.items():
        if value:
            print(f"✅ {key} is set")
            if key == "OPENAI_API_KEY":
                available_models.append("gpt-5")
            elif key == "ANTHROPIC_API_KEY":
                available_models.append("claude-sonnet-4-20250514")
            elif key == "GEMINI_API_KEY":
                available_models.append("gemini-2.5-pro")
        else:
            print(f"❌ {key} is not set")
    
    if not available_models:
        print("\n⚠️  No API keys found! Please set at least one API key:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export ANTHROPIC_API_KEY='your-key'")
        print("   export GEMINI_API_KEY='your-key'")
        sys.exit(1)
    
    print(f"🎯 Available models: {', '.join(available_models)}")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Define models to test (only those with API keys)
    requested_models = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]
    models = [model for model in requested_models if model in available_models]
    
    if not models:
        print("❌ None of the requested models are available with current API keys!")
        sys.exit(1)
    
    print(f"🎯 Testing {len(models)} models: {', '.join(models)}")
    
    if len(models) < len(requested_models):
        missing = set(requested_models) - set(models)
        print(f"⚠️  Skipping unavailable models: {', '.join(missing)}")
    
    # Load train data
    train_path = Path("../data/train.jsonl")
    if not train_path.exists():
        print(f"❌ Error: {train_path} not found!")
        sys.exit(1)
    
    records = load_train_data(train_path)
    if not records:
        print("❌ Error: No records found in train.jsonl!")
        sys.exit(1)
    
    # Select random subset
    selected_records = select_random_subset(records, n=3)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Save selected records as individual files
    saved_files = save_selected_records(selected_records, output_dir)
    
    # Run each model on each file
    print("\n" + "=" * 80)
    print("🚀 RUNNING MODELS")
    print("=" * 80)
    
    successful_runs = 0
    total_runs = len(models) * len(saved_files)
    
    for record_index, file_path in enumerate(saved_files, 1):
        print(f"\n📄 Processing Record {record_index}: {file_path.name}")
        print("-" * 60)
        
        for model in models:
            output_path = run_model_on_file(model, file_path, output_dir, record_index)
            if output_path:
                successful_runs += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 EXPLORATORY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"✅ Successful runs: {successful_runs}/{total_runs}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎲 Random seed: 42 (for reproducibility)")
    
    if successful_runs < total_runs:
        print(f"⚠️  {total_runs - successful_runs} runs failed - check API keys and model availability")
    
    print("\n📋 Generated files:")
    for file in sorted(output_dir.glob("*.md")):
        print(f"  - {file.name}")
    
    print("\n🎉 Exploratory analysis complete! Check the outputs to compare model performance.")


if __name__ == "__main__":
    main()
