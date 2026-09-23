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


def answer_question(docs, question):
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are a helpful study assistant.

Answer ONLY from the context provided below.

If the answer is not available in the context, reply exactly:

"I couldn't find the answer in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    llm = get_llm()
    response = llm.invoke(prompt)

    return response.content