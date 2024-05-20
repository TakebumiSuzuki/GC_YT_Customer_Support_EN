from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
from dotenv import load_dotenv
load_dotenv()

KOWLEDGE_FILE_DIR = "knowledge_EN_0508.json"


with open(KOWLEDGE_FILE_DIR, 'r', encoding='utf-8') as file:
    data = json.load(file)
print(len(data))
data = data[900:1000]
contents = [item['content'] for item in data if 'content' in item]
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="RETRIEVAL_DOCUMENT"
)
db = FAISS.from_texts(
    texts = contents,
    embedding = embeddings
)
db.save_local("EN_db_10")
print("終了")


# new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)

# docs = new_db.similarity_search("hi")
# print(docs)

# db1 = FAISS.load_local("testF1", embeddings, allow_dangerous_deserialization = True)
# db2 = FAISS.load_local("testF2", embeddings, allow_dangerous_deserialization = True)
# print('-----------')
# print(db1.docstore._dict)
# db1.merge_from(db2)
# print('-----------')
# print(db1.docstore._dict)

