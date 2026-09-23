import os
from dotenv import load_dotenv
from django.conf import settings
from langchain_groq import ChatGroq


load_dotenv(os.path.join(settings.BASE_DIR, ".env"))


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key
    )


def summarize_text(text):
    prompt = f"""
You are an AI Study Assistant.

Summarize the following study material.

Generate:

1. Overview
2. Important Topics
3. Key Concepts
4. Important Definitions
5. Conclusion

Study Material:
{text[:12000]}
"""

    llm = get_llm()
    response = llm.invoke(prompt)

    return response.content