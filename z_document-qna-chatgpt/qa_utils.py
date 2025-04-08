import os
import time
import openai
from fastapi import HTTPException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API Key
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required.")

# Initialize OpenAI Client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

def query_gpt(prompt_name: str, prompt: str, max_retries=3, retry_delay=5) -> str:
    """
    Query OpenAI GPT with automatic retry handling for rate limits.

    Args:
        prompt_name (str): Name of the prompt (for logging/debugging).
        prompt (str): The actual query to send to OpenAI.
        max_retries (int): Maximum number of retry attempts.
        retry_delay (int): Base delay for exponential backoff.

    Returns:
        str: The generated response from OpenAI.
    """
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are an AI assistant."}, {"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return response.choices[0].message.content
        
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(wait_time)
            else:
                raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded. Please try again later.")
        
        except openai.OpenAIError as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")
