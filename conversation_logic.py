import constants as K
import os
from dotenv import load_dotenv
from uuid import uuid4
load_dotenv()

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_google_genai import GoogleGenerativeAI
# from langchain_openai.chat_models import ChatOpenAI

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="RETRIEVAL_QUERY"
)

from langchain_community.vectorstores import FAISS
db = FAISS.load_local("en_0508_faiss.db", embeddings_model, allow_dangerous_deserialization = True)

# vectorstore = Chroma(
#     persist_directory = (
#         K.EN_VECSTORE if K.lang == "EN" else
#         K.JA_VECSTORE
#     ),
#     embedding_function = embeddings_model
# )

retriever = db.as_retriever(
    search_type = K.SEARCH_TYPE,
    search_kwargs = {'k': K.K, 'score_threshold': K.THRESH},
)

gemini = GoogleGenerativeAI(
        model = K.GEMINI_MODEL_NAME,
        google_api_key = os.getenv(K.GOOGLE_API_KEY)
)
print(gemini.invoke('Google Cloudでpythonを使ってtext embeddingをできるようにするには、何をインストールする必要がありますか'))

# llm = ChatOpenAI(
#     model = K.OPENAI_MODEL_NAME ,
#     temperature = K.OPENAI_TEMP,
# )

def retrieve(inputText, store):

    language = (
        "English" if K.lang == "EN" else
        "Japanese"
    )

    prompt_qustion = K.HYDE_PROMPT.format(language, language) + inputText
    print(f"-----------\nPROMPTQ: {prompt_qustion}")

    hyde_query = gemini.invoke(prompt_qustion)
    print("\n-------------\n")
    print(f"HyDE_Q: {hyde_query}")
    print("\n-------------\n")

    docs = retriever.invoke(hyde_query)

    source_text = ""
    for doc in docs:
        text = doc.page_content
        source_text += text + '\n---\n'
    print("get the docs through the retrieval")
    return source_text, docs



def invoke(inputText, docs, store):

    language = (
        "English" if K.lang == "EN" else
        "Japanese"
    )

    qa_system_prompt = K.QA_PROMPT

    qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            ("user", "{input}")]
            )

    question_answer_chain = create_stuff_documents_chain(gemini, qa_prompt)

    return question_answer_chain.invoke({"input": inputText, "context": docs, "language": language})








