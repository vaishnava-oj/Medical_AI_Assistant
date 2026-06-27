import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key from .env
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

def get_disease_explanation(disease):
    prompt = f"""
    Explain the disease '{disease}' in simple language.

    Include:
    1. What it is
    2. Common symptoms
    3. Causes
    4. Treatment
    5. Precautions
    6. Diet recommendations
    7. When to consult a doctor
    """

    response = model.generate_content(prompt)
    return response.text