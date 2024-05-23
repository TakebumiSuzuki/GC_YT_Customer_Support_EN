import constants as K
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


embeddings_model = GoogleGenerativeAIEmbeddings(
    model = K.EMBEDDING_MODEL_NAME,
    task_type = "RETRIEVAL_QUERY"
)

db = FAISS.load_local(K.EN_VECSTORE, embeddings_model, allow_dangerous_deserialization = True)

retriever = db.as_retriever(
    search_type = K.SEARCH_TYPE,
    search_kwargs = {'k': K.K, 'score_threshold': K.THRESH},
)

gemini = GoogleGenerativeAI(
        model = K.GEMINI_MODEL_NAME,
        google_api_key = os.getenv(K.GOOGLE_API_KEY),
)


def retrieve(inputText):
    language = ("English" if K.lang == "EN" else "Japanese")

    prompt_qustion = K.HYDE_PROMPT.format(language, language) + inputText

    hyde_query = gemini.invoke(prompt_qustion)
    print(f"HyDE_Q: {hyde_query}")

    docs = retriever.invoke(hyde_query)

    source_text = ""
    for doc in docs:
        text = doc.page_content
        source_text += text + '\n---\n'

    return source_text, docs


def get_stream(inputText, docs):
    language = ("English" if K.lang == "EN" else "Japanese")

    qa_system_prompt = K.QA_PROMPT

    qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            ("user", "{input}")]
    )

    question_answer_chain = create_stuff_documents_chain(gemini, qa_prompt) | StrOutputParser()

    return question_answer_chain.stream({
        "input": inputText,
        "context": docs,
        "language": language
        })



def error_handling(e):
    print(f"chunkのエラーです: {e}")










