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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("vector_store")
    return vector_store

# -----------------------
# QCM generation using Gemini through LangChain wrapper
# -----------------------
def generate_qcm_from_vectorstore(n_questions=10, k_chunks=8, difficulty="Moyen", model_name="gemini-2.0-flash"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error("Vector store introuvable. Upload et cliquez sur 'Submit' d'abord.")
        return []

    # retrieve top-k chunks
    docs = db.similarity_search("important points for making exam questions", k=k_chunks)
    context = "\n\n".join([d.page_content for d in docs])
    if not context.strip():
        st.error("Contexte insuffisant pour générer des QCM.")
        return []

    # prompt JSON strict
    prompt_template = PromptTemplate(
        input_variables=["context", "n", "difficulty"],
        template="""
    Tu es un créateur de QCM pour des étudiants basé uniquement sur le CONTEXTE fourni.
    Crée exactement {n} questions à choix multiple (4 choix : A, B, C, D) avec le niveau de difficulté {difficulty}.
    Chaque élément doit contenir : id (int), question (string), choices (array of 4 strings),
    answer (one of "A","B","C","D"), explanation (short, 1-2 phrases), source (fichier:page ou extrait).
    RETURNS: Strict JSON array ONLY (ne rien ajouter en dehors du JSON).

    Contexte:
    {context}
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    prompt_template= """  answer the question as detailed as possible from the provided context, make sure to provide all
    the details, if the answer is not in the provided context, just say "i am sorry , answer is not availble in this context"
    don't provide a wrong answer \n\n 
    context: {context} \n\n
    question: {question} \n\n Answer in English:"""

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    context = "\n\n".join([d.page_content for d in docs])
    response = chain.run({"context": context, "question": user_question})
    return response


# generate summary
def generate_summary_from_vectorstore(k_chunks=8, model_name="gemini-2.0-flash"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error("Vector store introuvable. Upload et cliquez sur 'Submit' d'abord.")
        return ""

    docs = db.similarity_search("important points for making a summary", k=k_chunks)
    context = "\n\n".join([d.page_content for d in docs])
    if not context.strip():
        st.error("Contexte insuffisant pour générer un résumé.")
        return ""

    prompt_template = PromptTemplate(
    input_variables=["context"],
    template="""
Tu es un assistant qui résume les documents PDF fournis. 
Fais un résumé **structuré** et **clair** du contenu suivant : 

{context}

Consignes :
- Utilise des grands titres (ex : 1. Introduction, 2. Concepts clés, 3. Applications, 4. Conclusion)
- Pour chaque grand titre, ajoute des sous-titres si nécessaire.
- Résume les informations importantes sous forme de phrases concises.
- Ne fais pas de JSON, juste un texte bien organisé avec titres et sous-titres.
- Sois synthétique mais complet.
"""
    )


    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    chain = LLMChain(llm=model, prompt=prompt_template)

    raw = chain.run({"context": context})
    summary = raw.strip()


    return summary




def generate_open_questions_from_vectorstore(n_questions=5, k_chunks=8, model_name="gemini-2.0-flash", difficulty="Moyen"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        st.error("Vector store introuvable. Upload et cliquez sur 'Submit' d'abord.")
        return []

    docs = db.similarity_search("important points for exam open questions", k=k_chunks)
    context = "\n\n".join([d.page_content for d in docs])
    if not context.strip():
        st.error("Contexte insuffisant pour générer des questions ouvertes.")
        return []

    prompt_template = PromptTemplate(
        input_variables=["context", "n", "difficulty"],
        template="""
Tu es un assistant qui génère des questions ouvertes pour préparer un examen.
Crée exactement {n} questions ouvertes basées uniquement sur le contexte fourni, 
avec un niveau de difficulté {difficulty}.
Retourne un JSON array avec chaque question sous la forme :
{{"id": 1, "question": "texte de la question", "model_answer": "réponse courte attendue"}}

Contexte : {context}
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

def validate_open_answer(user_ans, model_ans, question, model_name="gemini-2.0-flash"):
    """
    Utilise Gemini pour évaluer si la réponse utilisateur est correcte.
    Retourne True/False et un commentaire.
    """
    prompt_template = PromptTemplate(
        input_variables=["question", "model_ans", "user_ans"],
        template="""
            Tu es un correcteur automatique pour des questions ouvertes d'examen.
            Question: {question}
            Réponse attendue: {model_ans}
            Réponse de l'étudiant: {user_ans}

            Évalue si la réponse de l'étudiant est correcte.
            Réponds STRICTEMENT en JSON, uniquement avec des double quotes.
            Format exact attendu:
            {{"is_correct": true/false, "comment": "explication courte"}}
            Ne mets rien en dehors du JSON.
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
            result = {"is_correct": False, "comment": "Impossible d'évaluer"}
    else:
        result = {"is_correct": False, "comment": "Impossible d'évaluer"}
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
        st.title("Téléverse tes fichiers PDF pour créer un QCM/examen")
        pdf_docs = st.file_uploader("", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit PDFs"):
            if not pdf_docs:
                st.warning("Aucun PDF uploadé.")
            else:
                with st.spinner("Processing PDFs..."):
                    text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(text)
                    get_vector_store(chunks)
                    st.success("✅ PDF indexés et vector store sauvegardé.")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
        st.markdown("---")
        st.markdown("Utilise l'onglet **QCM** pour générer et passer un examen blanc.")

    tabs = st.tabs(["Chat (QA)", "QCM Generator"," Résumé", "Questions ouvertes"])

    # --- Chat Tab ---
    with tabs[0]:
        st.subheader("Chat (poser une question sur les PDFs indexés)")
        user_question = st.chat_input("Ask a question about the PDFs...")
        if user_question:
            with st.spinner("Processing..."):
                try:
                    response = run_qa_chain(user_question)
                    st.session_state.chat_history.append(("You", user_question))
                    st.session_state.chat_history.append(("Bot", response))
                except Exception as e:
                    st.error(f"Erreur QA: {e}")
        for role, text in st.session_state.chat_history:
            st.markdown(f"<div class='user-msg'>{text}</div>" if role=="You" else f"<div class='bot-msg'>{text}</div>", unsafe_allow_html=True)

    # --- QCM Tab ---
    with tabs[1]:
        st.subheader("Générateur de QCM")

        difficulty = st.selectbox(
            "Sélectionne le niveau de difficulté",
            ["Facile", "Moyen", "Difficile"]
        )

        num_q = st.number_input("Nombre de questions à générer", min_value=5, max_value=30, value=10, step=1)

        if st.button("Générer QCM depuis les PDFs indexés"):
            with st.spinner("Génération des questions..."):
                quiz = generate_qcm_from_vectorstore(n_questions=num_q, k_chunks=10, difficulty=difficulty)
                if quiz:
                    st.session_state.current_quiz = quiz
                    st.session_state.quiz_generated_at = time.time()
                    st.success(f"{len(quiz)} questions générées.")
                else:
                    st.error("Aucune question générée. Vérifie l'indexation ou le contenu des PDFs.")
        if st.session_state.current_quiz:
            quiz = st.session_state.current_quiz
            st.markdown(f"Questions prêtes : **{len(quiz)}**")
            for q in quiz:
                qid = q["id"]
                st.write(f"**Q{qid}. {q['question']}**")
                labeled_choices = [f"A. {q['choices'][0]}", f"B. {q['choices'][1]}", f"C. {q['choices'][2]}", f"D. {q['choices'][3]}"]
                st.radio("", labeled_choices, key=f"q_{qid}")
                if q.get("source"): st.markdown(f"<div class='small-muted'>Source: {q.get('source')}</div>", unsafe_allow_html=True)
                st.markdown("---")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Soumettre mes réponses"):
                    correct, total, pct, feedback = compute_score_and_feedback(quiz, st.session_state)
                    st.success(f"Tu as obtenu **{correct} / {total}** — **{pct}%**")
                    wrong = [f"Q{f['id']}" for f in feedback if not f['is_correct']]
                    if wrong: st.warning(f"Questions à revoir: {', '.join(wrong)}")
                    else: st.balloons()
                    st.session_state.last_feedback = feedback
            with col_b:
                if st.button("Voir les réponses (corrigé)"):
                    feedback = st.session_state.get("last_feedback", None)
                    if not feedback:
                        _, _, _, feedback = compute_score_and_feedback(quiz, st.session_state)
                    st.subheader("Corrigé complet")
                    for f in feedback:
                        st.write(f"**Q{f['id']}. {f['question']}**")
                        st.markdown(f"- **Réponse correcte**: {f['correct_answer_letter']} — {f['correct_answer_text']}")
                        st.markdown(f"- **Ton choix**: {f['your_answer']}")
                        if f['explanation']: st.markdown(f"- **Explication**: {f['explanation']}")
                        if f['source']: st.markdown(f"- **Source**: {f['source']}")
                        st.markdown("---")
            if st.download_button("⬇️ Télécharger le quiz (JSON)", json.dumps(quiz, ensure_ascii=False, indent=2), file_name="quiz.json"):
                st.success("Quiz téléchargé.")
    # --- Resume Tab ---
    with tabs[2]:
        st.subheader("Générateur de résumé à partir des PDFs indexés")

        if st.button("Générer Résumé"):
            with st.spinner("Génération du résumé..."):
                summary = generate_summary_from_vectorstore()
                if summary:
                    st.session_state.current_summary = summary
                    st.success("Résumé généré avec succès !")
                else:
                    st.error("Impossible de générer le résumé.")

        if "current_summary" in st.session_state:
            st.markdown(st.session_state.current_summary, unsafe_allow_html=True)

            if st.download_button("⬇️ Télécharger le résumé", st.session_state.current_summary, file_name="resume.txt"):
                st.success("Résumé téléchargé.")
    with tabs[3]:  # onglet "Questions ouvertes"
        st.subheader("Préparation examens : Questions ouvertes")

        num_open_q = st.number_input("Nombre de questions ouvertes à générer", min_value=3, max_value=20, value=5)
        difficulty_open = st.selectbox(
        "Sélectionne le niveau de difficulté pour les questions ouvertes",
        ["Facile", "Moyen", "Difficile"]
        )

        if st.button("Générer Questions ouvertes"):
            with st.spinner("Génération des questions..."):
                open_questions = generate_open_questions_from_vectorstore(
                    n_questions=num_open_q,
                    k_chunks=8,
                    model_name="gemini-2.0-flash",
                    difficulty=difficulty_open  # nouveau paramètre
                )
                if open_questions:
                    st.session_state.open_questions = open_questions
                    st.session_state.open_answers = {qid: "" for qid in [q["id"] for q in open_questions]}
                    st.success(f"{len(open_questions)} questions ouvertes générées ({difficulty_open}).")
                else:
                    st.error("Impossible de générer des questions ouvertes.")

        if "open_questions" in st.session_state:
            suffix = st.session_state.get("open_key_suffix", "")
            for q in st.session_state.open_questions:
                qid = q["id"]
                st.markdown(f"**Q{qid}. {q['question']}**")
                # Clé unique = id + suffix
                st.session_state.open_answers[qid] = st.text_area(
                    "Votre réponse :",
                    key=f"open_{qid}_{suffix}",
                    height=150,
                    value=""  # vide par défaut
                )
                st.markdown("---")
            if st.button("Soumettre mes réponses"):
                feedback_list = []
                for q in st.session_state.open_questions:
                    qid = q["id"]
                    user_ans = st.session_state.open_answers.get(qid, "").strip()
                    model_ans = q.get("model_answer", "").strip()

                    if not user_ans:
                        is_correct = False
                        comment = "Réponse vide"
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

                # score global
                total = len(feedback_list)
                correct = len([f for f in feedback_list if f["is_correct"]])
                pct = round((correct/total)*100) if total>0 else 0
                st.success(f"Score : {correct}/{total} — {pct}%")

                # feedback détaillé
                st.subheader("Feedback détaillé")
                for f in feedback_list:
                    st.write(f"**Q{f['id']}. {f['question']}**")
                    st.markdown(f"- **Votre réponse**: {f['your_answer']}")
                    st.markdown(f"- **Réponse attendue**: {f['model_answer']}")
                    st.markdown(f"- **Correct**: {'✅' if f['is_correct'] else '❌'}")
                    st.markdown(f"- **Commentaire**: {f['comment']}")
                    st.markdown("---")




if __name__ == "__main__":
    main()








































# import streamlit as st
# import os 
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.chains.question_answering import load_qa_chain
# from dotenv import load_dotenv


# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader=PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+=page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=1000)
#     chunks=text_splitter.split_text(text)
#     return chunks


# def get_vector_store(chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(chunks, embeddings)
#     vector_store.save_local("vector_store")
    

# def get_conversational_chain():
#     prompt_template= """  answer the question as detailed as possible from the provided context, make sure to provide all
#     the details, if the answer is not in the provided context, just say"i am sorry , answer is not availble in this context"
#     don't provide a wrong answer \n\n 
#     context: {context} \n\n
#     question: {question} \n\n Answer in English:"""

#     model=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.2)
#     prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain =load_qa_chain(llm=model, prompt=prompt, chain_type="stuff")
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db=FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
#     docs= new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response=chain(
#         {"input_documents": docs, "question": user_question},return_only_outputs=True
#     )
#     # print(response)
#     # st.write("",response["output_text"])

#     # Sauvegarder l'historique
#     st.session_state.chat_history.append(("You", user_question))
#     st.session_state.chat_history.append(("Bot", response["output_text"]))

#     return response["output_text"]










# def main():
#     st.set_page_config(page_title="Q/A Bot", layout="wide")
#     st.markdown("<h1 style='text-align:center;'>🤖 Q&A Bot</h1>", unsafe_allow_html=True)

#     # CSS pour bulles de chat
#     st.markdown("""
#         <style>
#         .chat-container { display: flex; flex-direction: column; }
#         .user-msg {
#             background-color: #e0e0e0;
#             padding: 10px 15px;
#             border-radius: 15px;
#             margin: 5px;
#             max-width: 85%;
#             align-self: flex-end;
#         }
#         .bot-msg {
#             background-color: #d1e7ff;
#             padding: 10px 15px;
#             border-radius: 15px;
#             margin: 5px;
#             max-width: 85%;
#             align-self: flex-start;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     # Initialiser l'historique
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     # Conteneur dynamique pour le chat
#     chat_container = st.container()

#     # Sidebar
#     with st.sidebar:
#         st.title("Upload PDF")
#         pdf_docs = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
        
#         if st.button("Submit"):
#             with st.spinner("Processing PDFs..."):
#                 text = get_pdf_text(pdf_docs)
#                 chunks = get_text_chunks(text)
#                 get_vector_store(chunks)
#                 st.success("✅ PDF uploaded successfully!")
#         if st.button("Clear Chat"):
#             st.session_state.chat_history = []

#     # Input utilisateur en bas
#     user_question = st.chat_input("Ask a question about the PDFs...")
#     if user_question:
#         with st.spinner("Processing..."):
#             user_input(user_question)


#     # Download chat in .txt
#     if st.sidebar.download_button("⬇️ Download Chat", "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history]),
#                                     file_name="chat_history.txt"):
#         st.success("Chat downloaded!")


#     # Affichage du chat dynamiquement
#     with chat_container:
#         for role, text in st.session_state.chat_history:
#             if role == "You":
#                 st.markdown(f"<div class='user-msg'>{text}</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown(f"<div class='bot-msg'>{text}</div>", unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()



























# def main():
#     st.set_page_config(page_title="Q/A")
#     st.header("Q&A Bot")

#     user_question = st.text_input("Ask a question about the PDF documents:")

#     with st.spinner("Processing..."):
#         if user_question:
#             user_input(user_question)

#         with st.sidebar:
#             st.title("Upload PDF")
#             pdf_docs = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
#             if st.button("submit"):
#                 text=get_pdf_text(pdf_docs)
#                 chunks=get_text_chunks(text)
#                 get_vector_store(chunks)
#                 st.success("PDF uploaded successfully.")
        
#          # Bouton pour vider l'historique
#     if st.button("Clear Chat"):
#         st.session_state.chat_history = []

#     st.set_page_config(page_title="Q/A", layout="wide")


        


# if __name__ == "__main__":
#     main()
