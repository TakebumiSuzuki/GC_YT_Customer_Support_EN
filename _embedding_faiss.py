################################################################################
#FAISSの最初のデータベースを作るための作業用スクリプト
#Google Embeddingはフリーだと1バッチあたり100リクエストしかできないようなので、
#上部のコードで、data[0:100]のようにして、100要素ずつembeddをしていく。
#kowledge_EN_0508の中には4562のテキストのチャンクがあるので、46回繰り返す。
#下部のコードで、merge_fromメソッドを使って、46個を一つのデータベースに集約していく。
#ちなみに、LangChainのバグと思われるが、GoogleGenerativeAIEmbeddingsの初期化の中の
#api_keyのパラメータはうまく動かない(ソースコードのsecrete key stringの部分がおかしいかと)
#従って、ターミナルで直接環境変数GOOGLE_API_KEYを入れてその状態でこのスクリプトを動かす
###############################################################################

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
from dotenv import load_dotenv
load_dotenv()

KOWLEDGE_FILE_DIR = "knowledge_EN_0508.json"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="RETRIEVAL_DOCUMENT"
)

####################上部####################################################

# with open(KOWLEDGE_FILE_DIR, 'r', encoding='utf-8') as file:
#     data = json.load(file)
# print(len(data))
# data = data[4500:4562]
# contents = [item['content'] for item in data if 'content' in item]

# db = FAISS.from_texts(
#     texts = contents,
#     embedding = embeddings
# )
# db.save_local("EN_db_46")
# print("終了")

####################下部#####################################################

base_db = FAISS.load_local("till_4000", embeddings, allow_dangerous_deserialization = True)
print(len(base_db.docstore._dict))
db = FAISS.load_local("EN_db_41", embeddings, allow_dangerous_deserialization = True)
base_db.merge_from(db)
db = FAISS.load_local("EN_db_42", embeddings, allow_dangerous_deserialization = True)
base_db.merge_from(db)
db = FAISS.load_local("EN_db_43", embeddings, allow_dangerous_deserialization = True)
base_db.merge_from(db)
db = FAISS.load_local("EN_db_44", embeddings, allow_dangerous_deserialization = True)
base_db.merge_from(db)
db = FAISS.load_local("EN_db_45", embeddings, allow_dangerous_deserialization = True)
base_db.merge_from(db)
db = FAISS.load_local("EN_db_46", embeddings, allow_dangerous_deserialization = True)
base_db.merge_from(db)


base_db.save_local("till_4562") #ここのセーブデ最終的なマージがファイルとして残る

print(len(base_db.docstore._dict))

