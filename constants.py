
lang = "EN"
# "JA" or "EN"

TAB_ICON = None

def TAB_TITLE(l):
    return (
        'YPP AI Assistant' if l == "EN" else
        'YPP AI アシスタント'
    )

def TITLE(l):
    return (
        'Partner Manager AI Support' if l == "EN" else
        'パートナーマネージャー AI サポート'
        )

def SUBTITLE(l):
    return (
        '☝ I can only answer information written in the Creator Support, which is public info. Please make sure to check the link to the information source in the left column and do not blindly accept AI-generated information.' if l == "EN" else
        '☝ YouTubeクリエーターサポートに書いている情報のみ答えられます。AIの情報を鵜呑みにせずに、左カラムの情報ソースのリンクを必ず確認して下さい。'
    )

def INPUT_HOLDER(l):
    return (
        'Ask me a question about YouTube!  I will not remember previous conversations.' if l == "EN" else
        'YouTubeに関する質問をして下さい。過去の会話は記憶しないことにご留意ください。'
    )

def CLEAR_BUTTON(l):
    return (
        'Clear' if l == "EN" else
        '会話をクリア'
    )

def SIDEBAR_SUBTITLE(l):
    return (
        '[ Resources ]' if l == "EN" else
        '[ 情報ソース ]'
    )


COHERE_API_KEY = 'COHERE_API_KEY'
COHERE_MODEL_NAME = 'command-r-plus'

GOOGLE_API_KEY = 'GOOGLE_API_KEY'
GEMINI_MODEL_NAME = 'models/gemini-1.5-pro-latest'

OPENAI_API_KEY = 'OPENAI_API_KEY'
OPENAI_MODEL_NAME = 'gpt-4o'
OPENAI_TEMP = 0.4

LANGSMITH_API_KEY = 'LANGSMITH_API_KEY'

EMBEDDING_MODEL_NAME = 'text-embedding-3-large'
EN_VECSTORE = 'data_en.chroma_db'
JA_VECSTORE = 'data_ja.chroma_db'
SEARCH_TYPE = 'similarity_score_threshold'
K = 4
FETCH_K = 15
THRESH = 0.30

CSS = """
    <style>
        header {visibility: hidden;}
        div[class^='block-container'] { padding-top: 2rem; }
    </style>
    """


REPHRASED_PROMPT = """
As an AI language model trained to improve query clarity, your task is to transform a YouTube creator's query into three rephrased sentences. Each sentence should enhance the original query's clarity and precision while preserving the creator's intent related to YouTube topics.

###
Instructions:
1. Rephrase the query to eliminate informal language or abbreviations and improve overall clarity.
2. Ensure that the each sentence maintain the original intention and provide a comprehensive refinement of the query.
3. Maintain a focus on enhancing the query's effectiveness for YouTube-related topics.
4. Respond in {}.
5. Ensure your response conforms to the two-sentence structure specified above.
6. Do NOT answer the question directly; instead, generate three rephrased sentences from the user's original question as guided.
###

Here is the original question from user:
{input}
"""


HYDE_PROMPT = """###
As an AI language model specializing in Hypothetical Document Embeddings (HyDE), your task is to transform questions from YouTube creators into clearer and more effective forms.

###
Instructions:
Do NOT answer the question directly: instead, generate three sentences from the user's original question in {} as guided below.
- For the first sentence: Reframe the question by formalizing any colloquial language or abbreviations to enhance clarity while preserving the original intent related to YouTube as much as possible.
- For the second and third sentences: Provide a contextually relevant explanation or answer that leverages your understanding of YouTube's functionalities.

###
Notes:
1. If the word "membership" appears in the question, it refers to a paid subscription service offered by YouTube channels to their viewers, known as channel membership.
2. If the term "premiere" appears, it refers to a feature that allows setting a future date and time for video publication.
3. If the word "shorts" appears, it refers to short videos.
4. Your responses should be in {}.
5. Do NOT answer the question directly

###
Example:
User's question: Hey, can I change or edit thumbnails of my short videos?
your output (*You write only three sentences): Can I edit the thumbnails for my short videos on YouTube? Yes, you can customize and edit thumbnails for your short videos on YouTube, just like you would for regular videos. This can be done by uploading an image that best represents your short video and will help it stand out to your viewers.

here is user's input:
"""



QA_PROMPT = """###
You are a YouTube customer support agent. Using the context provided below, answer the question as accurately as possible. If the context does not contain the information needed to answer, you MUST state that you do not know. Provide your response in up to four sentences and write in {language}.

###
Here is the context:
{context}
"""


