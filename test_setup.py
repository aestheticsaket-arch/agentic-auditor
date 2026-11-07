from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

print("1. Loading environment variables...")
load_dotenv()

print("2. Checking for API Key...")
if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_actual_key_here":
    print("ERROR: You need to put your real Groq API key in the .env file!")
    exit()

print("3. Connecting to Groq...")
# Using the guaranteed free-tier model with the correct prefix
llm = ChatGroq(
    model="openai/gpt-oss-20b", 
    temperature=0,
    max_retries=0, 
    timeout=15
)

print("4. Sending message to AI (Waiting for response)...")
try:
    response = llm.invoke("Say 'Environment setup successful!' if you can hear me.")
    print("\n--- AI RESPONSE ---")
    print(response.content)
except Exception as e:
    print(f"\nERROR OCCURRED: {e}")