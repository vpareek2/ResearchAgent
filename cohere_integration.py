import os
from dotenv import load_dotenv
import cohere

load_dotenv()
cohere_api_key = os.getenv('COHERE_API_KEY')

if not cohere_api_key:
    raise ValueError("Cohere API key is not set in the environment variables")

co = cohere.Client(cohere_api_key)

def summarize_text(text):
    if len(text) < 250:
        return text  # Return the original text if it's too short
    try:
        response = co.summarize(text=text, length='short')
        return response.summary
    except Exception as e:
        print(f"Error in summarizing text: {e}")
        return text  # Return the original text in case of any error