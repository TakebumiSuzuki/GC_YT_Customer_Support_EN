# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import constants as K
import streamlit as st
import conversation_logic as logic
from PIL import Image
st.set_page_config(
     page_title = K.TAB_TITLE(K.lang),
     page_icon = Image.open("./images/bear_icon_fabicon.jpeg"),
     layout = "wide",
     initial_sidebar_state = "expanded"
)

ss = st.session_state

if "store" not in ss:
    ss["store"] = []
message_list = ss["store"]



st.markdown(K.CSS, unsafe_allow_html=True)

st.title(K.TITLE(K.lang))
st.write(K.SUBTITLE(K.lang))

def setAvatar(role):
    if role == "AI": return "./images/bear_icon_avator.jpeg"
    else: return None

# Display chat messages from history on app rerun
if message_list != []:
    for message in message_list:
        with st.chat_message(message["role"],avatar=setAvatar(message["role"])):
            st.markdown(message["content"])
            # if message["role"] == "AI":
            #     st.button('docs履歴')

    if "show_button" in ss and ss["show_button"] == True:
        clear_button = st.button(K.CLEAR_BUTTON(K.lang))
        if clear_button == True:
            ss["store"] = []
            ss["retrived_text"] = ""
            clear_button = False
            st.rerun()

def delete_button():
    ss["show_button"] = False

# Accept user input
if input := st.chat_input(K.INPUT_HOLDER(K.lang), on_submit = delete_button):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(input)

    retrieved = logic.retrieve(input, ss["store"])
    ss["retrived_text"] = retrieved[0]
    logic.invoke(input, retrieved[1], ss["store"])

    ss["show_button"] = True
    st.rerun()

with st.sidebar:
    st.subheader(K.SIDEBAR_SUBTITLE(K.lang))
    if "retrived_text" in ss:
        st.markdown(ss["retrived_text"])



