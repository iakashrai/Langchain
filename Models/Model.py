from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.llms import GoogleGenerativeAI

llm = GoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

response = llm.invoke("What is the capital of India?")

print(response)
