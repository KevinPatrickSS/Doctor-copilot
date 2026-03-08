import os
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
assert api_key, "GEMINI_API_KEY not set"

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash-lite")

response = model.generate_content(
    "Say 'Gemini API is working' and nothing else."
)

print(response.text)
