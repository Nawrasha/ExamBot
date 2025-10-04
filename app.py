import streamlit as st
import os 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import json
import re
import time
import uuid
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------
# Utilities for PDF -> chunks -> vector store
# -----------------------
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        text += f"\n\n--- FILE: {getattr(pdf, 'name', 'uploaded_pdf')} ---\n\n"
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            text += f"\n\n--PAGE {i+1}--\n{page_text}\n"
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=500)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("vector_store")
    return vector_store

# -----------------------
# QCM generation using Gemini through LangChain wrapper
# -----------------------
def generate_qcm_from_vectorstore(n_questions=10, k_chunks=8, difficulty="Medium", model_name="gemini-2.0-flash-exp"):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
    try:
        db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error("Vector store not found. Please upload PDFs and click 'Submit' first.")
        return []

    # retrieve top-k chunks
    docs = db.similarity_search("important points for making exam questions", k=k_chunks)
    context = "\n\n".join([d.page_content for d in docs])
    if not context.strip():
        st.error("Insufficient context to generate MCQs.")
        return []

    # prompt JSON strict
    prompt_template = PromptTemplate(
        input_variables=["context", "n", "difficulty"],
        template="""
    You are an MCQ creator for students, using ONLY the provided CONTEXT.
    Create exactly {n} multiple-choice questions (4 choices: A, B, C, D) with difficulty level {difficulty}.
    Each item must contain: id (int), question (string), choices (array of 4 strings),
    answer (one of "A","B","C","D"), explanation (short, 1-2 sentences), source (file:page or snippet).
    RETURNS: Strict JSON array ONLY (do not add anything outside JSON).

    Context:
    {context}
    Answer in English.
    """
    )
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    chain = LLMChain(llm=model, prompt=prompt_template)
    raw = chain.run({"context": context, "n": n_questions, "difficulty": difficulty})

    # parse JSON
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\[.*\]", raw, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = []
        else:
            data = []
    # validate
    valid = []
    for item in data:
        if all(k in item for k in ("id","question","choices","answer","explanation")) and len(item["choices"])==4:
            valid.append(item)
    return valid

# -----------------------
# Helper: compute score (corrected)
# -----------------------
def compute_score_and_feedback(quiz, session_state):
    total = len(quiz)
    correct = 0
    feedback_list = []
    for q in quiz:
        qid = q["id"]
        key = f"q_{qid}"
        user_sel = session_state.get(key, None)
        # normalize: remove letter prefix if exists
        if isinstance(user_sel, str) and len(user_sel) > 3 and user_sel[1:3] == ". ":
            user_text = user_sel[3:].strip()
            user_letter = user_sel[0].upper()
        else:
            user_text = str(user_sel).strip() if user_sel else ""
            user_letter = user_sel.strip().upper() if user_sel else ""

        # correct answer
        correct_letter = q["answer"].strip().upper()
        letter_to_idx = {"A":0,"B":1,"C":2,"D":3}
        correct_idx = letter_to_idx.get(correct_letter, 0)
        correct_choice_text = q["choices"][correct_idx]

        # check if correct: either letter or text matches
        is_correct = (user_letter == correct_letter) or (user_text == correct_choice_text)
        if is_correct:
            correct += 1

        feedback_list.append({
            "id": qid,
            "question": q["question"],
            "your_answer": user_sel,
            "correct_answer_letter": correct_letter,
            "correct_answer_text": correct_choice_text,
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
            "source": q.get("source", "")
        })
    score_pct = round((correct/total)*100) if total>0 else 0
    return correct, total, score_pct, feedback_list

# -----------------------
# QA chain for Tab1
# -----------------------
def run_qa_chain(user_question):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
    new_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    prompt_template= """  answer the question as detailed as possible from the provided context, make sure to provide all
    the details, if the answer is not in the provided context, just say "i am sorry , answer is not availble in this context"
    don't provide a wrong answer \n\n 
    context: {context} \n\n
    question: {question} \n\n Answer in English:"""

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    context = "\n\n".join([d.page_content for d in docs])
    response = chain.run({"context": context, "question": user_question})
    return response


# generate summary
def generate_summary_from_vectorstore(k_chunks=8, model_name="gemini-2.0-flash-exp"):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
    try:
        db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error("Vector store not found. Please upload PDFs and click 'Submit' first.")
        return ""

    docs = db.similarity_search("important points for making a summary", k=k_chunks)
    context = "\n\n".join([d.page_content for d in docs])
    if not context.strip():
        st.error("Insufficient context to generate a summary.")
        return ""

    prompt_template = PromptTemplate(
    input_variables=["context"],
    template="""
You are an assistant that summarizes the provided PDF documents.
Write a structured and clear summary of the following content:

{context}

Guidelines:
- Use major headings (e.g., 1. Introduction, 2. Key Concepts, 3. Applications, 4. Conclusion)
- Add subheadings when helpful.
- Summarize important information in concise sentences.
- Do NOT output JSON; produce well-organized text with headings and subheadings.
- Be concise yet complete.

Answer in English.
"""
    )


    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    chain = LLMChain(llm=model, prompt=prompt_template)

    raw = chain.run({"context": context})
    summary = raw.strip()


    return summary




def generate_open_questions_from_vectorstore(n_questions=5, k_chunks=8, model_name="gemini-2.0-flash-exp", difficulty="Medium"):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
    try:
        db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        st.error("Vector store not found. Please upload PDFs and click 'Submit' first.")
        return []

    docs = db.similarity_search("important points for exam open questions", k=k_chunks)
    context = "\n\n".join([d.page_content for d in docs])
    if not context.strip():
        st.error("Insufficient context to generate open questions.")
        return []

    prompt_template = PromptTemplate(
        input_variables=["context", "n", "difficulty"],
        template="""
You are an assistant that generates open-ended exam questions.
Create exactly {n} open questions based ONLY on the provided context,
with a difficulty level of {difficulty}.
Return a JSON array where each item is of the form:
{{"id": 1, "question": "question text", "model_answer": "short expected answer"}}

Context: {context}

Answer in English.
"""
    )

    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    chain = LLMChain(llm=model, prompt=prompt_template)
    raw = chain.run({"context": context, "n": n_questions, "difficulty": difficulty})

    try:
        questions = json.loads(raw)
    except Exception:
        m = re.search(r"\[.*\]", raw, re.S)
        questions = json.loads(m.group(0)) if m else []

    return questions


# évaluation des réponses ouvertes

def validate_open_answer(user_ans, model_ans, question, model_name="gemini-2.0-flash-exp"):
    """
    Utilise Gemini pour évaluer si la réponse utilisateur est correcte.
    Retourne True/False et un commentaire.
    """
    prompt_template = PromptTemplate(
        input_variables=["question", "model_ans", "user_ans"],
        template="""
            You are an automatic grader for open-ended exam questions.
            Question: {question}
            Expected answer: {model_ans}
            Student answer: {user_ans}

            Evaluate whether the student's answer is correct.
            Respond STRICTLY in JSON, using only double quotes.
            Exact format:
            {{"is_correct": true/false, "comment": "short explanation"}}
            Do not include anything outside JSON.
            Answer in English.
            """
            )
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    chain = LLMChain(llm=model, prompt=prompt_template)
    raw = chain.run({"question": question, "model_ans": model_ans, "user_ans": user_ans})

    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            result = json.loads(m.group(0))
        except:
            result = {"is_correct": False, "comment": "Unable to evaluate"}
    else:
        result = {"is_correct": False, "comment": "Unable to evaluate"}
    return result["is_correct"], result["comment"]




# -----------------------
# Main UI
# -----------------------
def main():
    st.set_page_config(page_title="ExamBot", layout="wide")
    st.markdown("<h1 style='text-align:center;'>🤖 ExamBot </h1>", unsafe_allow_html=True)

    st.markdown("""
        <style>
        .chat-container { display: flex; flex-direction: column; }
        .user-msg { background-color: #e0e0e0; padding: 10px 15px; border-radius: 15px; margin: 5px; max-width: 85%; align-self: flex-end; }
        .bot-msg { background-color: #d1e7ff; padding: 10px 15px; border-radius: 15px; margin: 5px; max-width: 85%; align-self: flex-start; }
        .small-muted { font-size:12px; color: #666; }
        </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "current_quiz" not in st.session_state: st.session_state.current_quiz = []
    if "quiz_generated_at" not in st.session_state: st.session_state.quiz_generated_at = None

    with st.sidebar:
        st.title("Upload your PDF files to create an exam/MCQ")
        pdf_docs = st.file_uploader("Load PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit PDFs"):
            if not pdf_docs:
                st.warning("No PDF uploaded.")
            else:
                with st.spinner("Processing PDFs..."):
                    text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(text)
                    get_vector_store(chunks)
                    st.success("✅ PDFs indexed and vector store saved.")
        st.markdown("---")
        st.markdown("Use the **MCQ** (Multiple Choice Questions) tab to generate and take a mock exam.")

    tabs = st.tabs(["Chat (QA)", "MCQ Generator"," Summary", "Open Questions"])

    # --- Chat Tab ---
    with tabs[0]:
        st.subheader("Chat (ask a question about the indexed PDFs)")
        user_question = st.chat_input("Ask a question about the PDFs...")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
        if user_question:
            with st.spinner("Processing..."):
                try:
                    response = run_qa_chain(user_question)
                    st.session_state.chat_history.append(("You", user_question))
                    st.session_state.chat_history.append(("Bot", response))
                except Exception as e:
                    st.error(f"QA error: {e}")
        for role, text in st.session_state.chat_history:
            st.markdown(f"<div class='user-msg'>{text}</div>" if role=="You" else f"<div class='bot-msg'>{text}</div>", unsafe_allow_html=True)
    # --- QCM Tab ---
    with tabs[1]:
        st.subheader("MCQ Generator")

        difficulty = st.selectbox(
            "Select difficulty level",
            ["Easy", "Medium", "Hard"]
        )

        num_q = st.number_input("Number of questions to generate", min_value=5, max_value=30, value=10, step=1)

        if st.button("Generate MCQs from indexed PDFs"):
            with st.spinner("Generating questions..."):
                quiz = generate_qcm_from_vectorstore(n_questions=num_q, k_chunks=10, difficulty=difficulty)
                if quiz:
                    st.session_state.current_quiz = quiz
                    st.session_state.quiz_generated_at = time.time()
                    st.success(f"{len(quiz)} questions generated.")
                else:
                    st.error("No questions generated. Check indexing or PDF content.")
        if st.session_state.current_quiz:
            quiz = st.session_state.current_quiz
            st.markdown(f"Questions ready: **{len(quiz)}**")
            for q in quiz:
                qid = q["id"]
                st.write(f"**Q{qid}. {q['question']}**")
                labeled_choices = [f"A. {q['choices'][0]}", f"B. {q['choices'][1]}", f"C. {q['choices'][2]}", f"D. {q['choices'][3]}"]
                st.radio("", labeled_choices, key=f"q_{qid}")
                if q.get("source"): st.markdown(f"<div class='small-muted'>Source: {q.get('source')}</div>", unsafe_allow_html=True)
                st.markdown("---")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Submit my answers"):
                    correct, total, pct, feedback = compute_score_and_feedback(quiz, st.session_state)
                    st.success(f"You scored **{correct} / {total}** — **{pct}%**")
                    wrong = [f"Q{f['id']}" for f in feedback if not f['is_correct']]
                    if wrong: st.warning(f"Questions to review: {', '.join(wrong)}")
                    else: st.balloons()
                    st.session_state.last_feedback = feedback
            with col_b:
                if st.button("View answers (solution)"):
                    feedback = st.session_state.get("last_feedback", None)
                    if not feedback:
                        _, _, _, feedback = compute_score_and_feedback(quiz, st.session_state)
                    st.subheader("Full solution")
                    for f in feedback:
                        st.write(f"**Q{f['id']}. {f['question']}**")
                        st.markdown(f"- **Correct answer**: {f['correct_answer_letter']} — {f['correct_answer_text']}")
                        st.markdown(f"- **Your choice**: {f['your_answer']}")
                        if f['explanation']: st.markdown(f"- **Explanation**: {f['explanation']}")
                        if f['source']: st.markdown(f"- **Source**: {f['source']}")
                        st.markdown("---")
            if st.download_button("⬇️ Download quiz (JSON)", json.dumps(quiz, ensure_ascii=False, indent=2), file_name="quiz.json"):
                st.success("Quiz downloaded.")
    # --- Resume Tab ---
    with tabs[2]:
        st.subheader("Summary generator from indexed PDFs")

        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                summary = generate_summary_from_vectorstore()
                if summary:
                    st.session_state.current_summary = summary
                    st.success("Summary generated successfully!")
                else:
                    st.error("Unable to generate the summary.")

        if "current_summary" in st.session_state:
            st.markdown(st.session_state.current_summary, unsafe_allow_html=True)

            if st.download_button("⬇️ Download summary", st.session_state.current_summary, file_name="summary.txt"):
                st.success("Summary downloaded.")
    with tabs[3]:  # Open questions tab
        st.subheader("Exam preparation: Open Questions")

        num_open_q = st.number_input("Number of open questions to generate", min_value=3, max_value=20, value=5)
        difficulty_open = st.selectbox(
        "Select difficulty level for open questions",
        ["Easy", "Medium", "Hard"]
        )

        if st.button("Generate Open Questions", key="generate_open_questions"):
            with st.spinner("Generating questions..."):
                open_questions = generate_open_questions_from_vectorstore(
                    n_questions=num_open_q,
                    k_chunks=8,
                    model_name="gemini-2.0-flash-exp",
                    difficulty=difficulty_open
                )
                if open_questions:
                    st.session_state.open_questions = open_questions
                    st.session_state.open_answers = {qid: "" for qid in [q["id"] for q in open_questions]}
                    # Force new widget keys so previous text areas don't keep old values
                    st.session_state.open_key_suffix = str(uuid.uuid4())
                    st.success(f"{len(open_questions)} open questions generated ({difficulty_open}).")
                else:
                    st.error("Unable to generate open questions.")

        if "open_questions" in st.session_state:
            suffix = st.session_state.get("open_key_suffix", "")
            for q in st.session_state.open_questions:
                qid = q["id"]
                st.markdown(f"**Q{qid}. {q['question']}**")
                # Unique key = id + suffix
                st.session_state.open_answers[qid] = st.text_area(
                    "Your answer:",
                    key=f"open_{qid}_{suffix}",
                    height=150,
                    value=st.session_state.open_answers.get(qid, "")
                )
                st.markdown("---")
            if st.button("Submit my answers", key="submit_open_answers"):
                feedback_list = []
                for q in st.session_state.open_questions:
                    qid = q["id"]
                    user_ans = st.session_state.open_answers.get(qid, "").strip()
                    model_ans = q.get("model_answer", "").strip()

                    if not user_ans:
                        is_correct = False
                        comment = "Empty answer"
                    else:
                        is_correct, comment = validate_open_answer(user_ans, model_ans, q["question"])

                    feedback_list.append({
                        "id": qid,
                        "question": q["question"],
                        "your_answer": user_ans,
                        "model_answer": model_ans,
                        "is_correct": is_correct,
                        "comment": comment
                    })

                # overall score
                total = len(feedback_list)
                correct = len([f for f in feedback_list if f["is_correct"]])
                pct = round((correct/total)*100) if total>0 else 0
                st.success(f"Score: {correct}/{total} — {pct}%")

                # detailed feedback
                st.subheader("Detailed feedback")
                for f in feedback_list:
                    st.write(f"**Q{f['id']}. {f['question']}**")
                    st.markdown(f"- **Your answer**: {f['your_answer']}")
                    st.markdown(f"- **Expected answer**: {f['model_answer']}")
                    st.markdown(f"- **Correct**: {'✅' if f['is_correct'] else '❌'}")
                    st.markdown(f"- **Comment**: {f['comment']}")
                    st.markdown("---")




if __name__ == "__main__":
    main()

