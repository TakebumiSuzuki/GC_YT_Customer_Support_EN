import constants as K
import streamlit as st
import conversation_logic as logic
from PIL import Image
import time
import random

st.set_page_config(
     page_title = K.TAB_TITLE(K.lang),
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

# Display chat messages from history on app rerun
if message_list != []:
    for message in message_list:
        with st.chat_message(message["role"]):
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

    #ユーザーからのインプットをそのままUIに表示する
    with st.chat_message("user"):
        st.markdown(input)
    #ユーザーからのインプットをsession_stateに記録
    ss["store"].append({"role" : "user", "content" : input})


    #AIからの返答をストリームにて取得、表示する
    with st.chat_message("AI"):
        msg_holder = st.empty()
        msg_holder.markdown("Searching...")

        retrieved = logic.retrieve(input)
        ss["retrived_text"] = retrieved[0]

        with st.sidebar:
            st.subheader(K.SIDEBAR_SUBTITLE(K.lang))
            if "retrived_text" in ss:
                st.markdown(ss["retrived_text"])

        msg_holder.markdown("Reading...")
        full_response = ""
        stream = logic.get_stream(input, retrieved[1])
        for chunk in stream:
            try:
                word_count = 0
                random_int = random.randint(5, 10)
                for word in chunk:
                    full_response += word
                    word_count += 1
                    if word_count == random_int:
                        time.sleep(0.05)
                        msg_holder.markdown(full_response + "_")
                        word_count = 0
                        random_int = random.randint(5, 10)
            except Exception as e:
                logic.error_handling(e)

        msg_holder.markdown(full_response)

    #AIからの返答をsession_stateに記録
    ss["store"].append({"role" : "AI", "content" : full_response})
    ss["show_button"] = True
    st.rerun()


else:
    with st.sidebar:
        st.subheader(K.SIDEBAR_SUBTITLE(K.lang))
        if "retrived_text" in ss:
            st.markdown(ss["retrived_text"])




