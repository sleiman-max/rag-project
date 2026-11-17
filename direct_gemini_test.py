from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options=types.HttpOptions(api_version="v1beta")
)

print("Sending request to Gemini...")

response = client.models.generate_content(
    model="gemini-flash-lite-latest",
    contents="Reply with: Gemini is working!"
)

print("Gemini response:", response.text)