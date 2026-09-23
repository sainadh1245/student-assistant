import os

from django.conf import settings
from django.shortcuts import render
from dotenv import load_dotenv

from .utils.pdf_reader import extract_text
from .utils.vector_store import (
    create_vector_store,
    load_vector_store
)
from .utils.rag_chat import answer_question
from .utils.summarizer import summarize_text


load_dotenv()


def home(request):
    summary = request.session.get("summary", "")
    answer = ""
    uploaded_file = request.session.get("uploaded_file", "")

    if not request.session.session_key:
        request.session.create()

    session_id = request.session.session_key

    index_path = os.path.join(
        settings.BASE_DIR,
        "faiss_indexes",
        session_id
    )

    os.makedirs(index_path, exist_ok=True)

    if request.method == "POST":

        # PDF / Word Document Upload
        if "document" in request.FILES:

            uploaded_document = request.FILES["document"]

            file_name = uploaded_document.name.lower()

            allowed_extensions = (
                ".pdf",
                ".docx"
            )

            if file_name.endswith(allowed_extensions):

                media_folder = os.path.join(
                    settings.MEDIA_ROOT,
                    session_id
                )

                os.makedirs(
                    media_folder,
                    exist_ok=True
                )

                file_path = os.path.join(
                    media_folder,
                    uploaded_document.name
                )

                with open(
                    file_path,
                    "wb+"
                ) as destination:

                    for chunk in uploaded_document.chunks():
                        destination.write(chunk)

                # Extract text from PDF / DOCX
                text = extract_text(file_path)

                if text.strip():

                    # Generate summary
                    summary = summarize_text(text)

                    # Create FAISS vector database
                    create_vector_store(
                        text,
                        index_path
                    )

                    # Save session information
                    request.session["summary"] = summary
                    request.session["uploaded_file"] = (
                        uploaded_document.name
                    )

                    uploaded_file = uploaded_document.name

        # Question Answering
        question = request.POST.get(
            "question",
            ""
        ).strip()

        if question and os.path.exists(index_path):

            try:

                db = load_vector_store(
                    index_path
                )

                docs = db.similarity_search(
                    question,
                    k=4
                )

                answer = answer_question(
                    docs,
                    question
                )

            except Exception as e:

                answer = (
                    f"Error while generating answer: {str(e)}"
                )

    return render(
        request,
        "study_assistant/index.html",
        {
            "summary": summary,
            "answer": answer,
            "uploaded_file": uploaded_file
        }
    )