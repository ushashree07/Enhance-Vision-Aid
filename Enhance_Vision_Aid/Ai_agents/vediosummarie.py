"""
Video Summarization Module
===========================
This module provides video/image analysis and summarization using Google's
Gemini AI model. It processes images and generates detailed descriptions
using state-of-the-art vision language models.

Features:
    - Image file upload and processing
    - Content generation using Gemini 2.0 Flash model
    - Streamlit integration for web UI
    - DuckDuckGo integration for web search
    
Configuration:
    - Requires GEMINI_API_KEY in environment variables
    - Uses Google generative AI SDK for model access
    
Dependencies:
    - streamlit: Web UI framework
    - google-generativeai: Google Gemini API client
    - phi: AI framework for agents
    
Usage:
    streamlit run vediosummarie.py
    
Author: Enhance Vision Aid Team
"""

import streamlit as st 
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv

load_dotenv()

import os
from google import genai

def analyze_image_with_gemini(image_path, query=None):
    """
    Analyze an image using Google's Gemini vision model.
    
    Args:
        image_path (str): Path to the image file to analyze
        query (str, optional): Specific question or analysis prompt for the image.
                              Defaults to "What can you tell me about these instruments?"
    
    Returns:
        str: Generated text description/analysis of the image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        KeyError: If GEMINI_API_KEY is not set in environment
        google.api_core.exceptions.GoogleAPICallError: If API call fails
        
    Example:
        >>> result = analyze_image_with_gemini("image.jpg", "Describe this scene")
        >>> print(result)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise KeyError("GEMINI_API_KEY not found in environment variables")
    
    if not query:
        query = "What can you tell me about these instruments?"
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
        # Upload and process image
        file_ref = client.files.upload(file=image_path)
        print(f'File uploaded: {file_ref}')
        
        # Generate content using Gemini 2.0 Flash model
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[query, file_ref]
        )
        
        return response.text
        
    except Exception as e:
        raise Exception(f"Error analyzing image: {str(e)}")


# Example usage with direct execution
if __name__ == "__main__":
    try:
        img_path = "F:/Enhanced_Vision_Aid/Testing-images/istockphoto-656497862-612x612.jpg"
        result = analyze_image_with_gemini(img_path)
        print("Analysis Result:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
