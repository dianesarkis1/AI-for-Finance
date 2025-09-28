#!/usr/bin/env python3
"""
Legal Document to Executive Summary Generator
A GenAI project for creating memo pages from legal documents
"""

# ============================================================================
# STEP 1: LOAD ALL REQUIRED PACKAGES AND SETUP
# ============================================================================

import os
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Core data processing
import pandas as pd
import numpy as np

# Document processing
from pathlib import Path
import PyPDF2
import docx
import fitz  # PyMuPDF for advanced PDF processing

# LLM and AI packages
import openai
from openai import OpenAI
import anthropic
from anthropic import Anthropic

# Alternative LLM options
import requests
import time

# Text processing and NLP
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# Configuration and environment
from dotenv import load_dotenv
import yaml

# ============================================================================
# CONFIGURATION SETUP
# ============================================================================

class LegalDocumentProcessor:
    """
    Main class for processing legal documents and generating executive summaries
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the legal document processor with all required packages
        
        Args:
            config_path: Path to configuration file (optional)
        """
        print("🚀 Initializing Legal Document Processor...")
        
        # Load environment variables
        self._load_environment()
        
        # Initialize LLM clients
        self._setup_llm_clients()
        
        # Setup NLP tools
        self._setup_nlp_tools()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        self._setup_logging()
        
        print("✅ All packages and tools initialized successfully!")
    
    def _load_environment(self):
        """
        Load environment variables and API keys
        """
        print("📋 Loading environment variables...")
        
        # Load .env file if it exists
        load_dotenv()
        
        # Store API keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Validate API keys
        if not self.openai_api_key and not self.anthropic_api_key:
            print("⚠️  Warning: No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            print("   You can create a .env file with your API keys")
    
    def _setup_llm_clients(self):
        """
        Initialize LLM clients for different providers
        """
        print("🤖 Setting up LLM clients...")
        
        # OpenAI client
        if self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                print("✅ OpenAI client initialized")
            except Exception as e:
                print(f"❌ Error initializing OpenAI client: {e}")
                self.openai_client = None
        else:
            self.openai_client = None
        
        # Anthropic client
        if self.anthropic_api_key:
            try:
                self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
                print("✅ Anthropic client initialized")
            except Exception as e:
                print(f"❌ Error initializing Anthropic client: {e}")
                self.anthropic_client = None
        else:
            self.anthropic_client = None
    
    def _setup_nlp_tools(self):
        """
        Setup NLP tools for text processing
        """
        print("📝 Setting up NLP tools...")
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            print("✅ NLTK data downloaded")
        except Exception as e:
            print(f"⚠️  Warning: Could not download NLTK data: {e}")
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        except OSError:
            print("⚠️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration settings
        """
        default_config = {
            "llm_settings": {
                "preferred_provider": "openai",  # or "anthropic"
                "model": "gpt-4",  # or "claude-3-sonnet-20240229"
                "max_tokens": 4000,
                "temperature": 0.1
            },
            "document_processing": {
                "supported_formats": [".pdf", ".docx", ".txt"],
                "max_file_size_mb": 50,
                "extract_tables": True,
                "extract_images": False
            },
            "summary_settings": {
                "max_summary_length": 1000,
                "include_key_metrics": True,
                "include_risk_assessment": True,
                "format_output": "markdown"  # or "html", "plain_text"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                # Merge with default config
                default_config.update(user_config)
                print(f"✅ Configuration loaded from {config_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self):
        """
        Setup logging for the application
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('legal_processor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        print("✅ Logging setup complete")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get list of available models from configured providers
        """
        available_models = {}
        
        if self.openai_client:
            try:
                models = self.openai_client.models.list()
                available_models["openai"] = [model.id for model in models.data]
                print("✅ OpenAI models retrieved")
            except Exception as e:
                print(f"❌ Error retrieving OpenAI models: {e}")
        
        if self.anthropic_client:
            # Anthropic has a fixed set of models
            available_models["anthropic"] = [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229", 
                "claude-3-haiku-20240307"
            ]
            print("✅ Anthropic models available")
        
        return available_models
    
    def test_connection(self) -> Dict[str, bool]:
        """
        Test connection to LLM providers
        """
        results = {}
        
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                results["openai"] = True
                print("✅ OpenAI connection successful")
            except Exception as e:
                results["openai"] = False
                print(f"❌ OpenAI connection failed: {e}")
        
        if self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                results["anthropic"] = True
                print("✅ Anthropic connection successful")
            except Exception as e:
                results["anthropic"] = False
                print(f"❌ Anthropic connection failed: {e}")
        
        return results

def main():
    """
    Main function - test the setup
    """
    print("=" * 60)
    print("LEGAL DOCUMENT TO EXECUTIVE SUMMARY GENERATOR")
    print("=" * 60)
    
    # Initialize the processor
    processor = LegalDocumentProcessor()
    
    # Test connections
    print("\n🔗 Testing LLM connections...")
    connection_results = processor.test_connection()
    
    # Show available models
    print("\n📋 Available models:")
    available_models = processor.get_available_models()
    for provider, models in available_models.items():
        print(f"  {provider.upper()}: {', '.join(models[:3])}...")
    
    print("\n✅ Setup complete! Ready to process legal documents.")
    print("\nNext steps:")
    print("1. Add your API keys to a .env file")
    print("2. Install missing packages: pip install -r requirements.txt")
    print("3. Create the document loading function")
    print("4. Implement the summary generation logic")

if __name__ == "__main__":
    main()
