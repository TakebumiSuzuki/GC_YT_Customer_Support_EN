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

    with st.chat_message("user"):
        st.markdown(input)
    ss["store"].append({"role" : "user", "content" : input})

    full_response = ""
    with st.chat_message("AI"):

        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        retrieved = logic.retrieve(input, ss["store"])
        ss["retrived_text"] = retrieved[0]

        stream = logic.get_stream(input, retrieved[1], ss["store"])

        # try:
        #     for chunk in stream:
        #         word_count = 0
        #         random_int = random.randint(5,10)
        #         for word in chunk.text:
        #             full_response+=word
        #             word_count+=1
        #             if word_count == random_int:
        #                 time.sleep(0.05)
        #                 message_placeholder.markdown(full_response + "_")
        #                 word_count = 0
        #                 random_int = random.randint(5,10)
        #     message_placeholder.write_stream(full_response)

        # except Exception as e:
        #     st.exception(e)

        # message_placeholder.markdown(stream)

        final_response = message_placeholder.write_stream(stream)

    ss["store"].append({"role" : "AI", "content" : final_response})
    ss["show_button"] = True
    st.rerun()


with st.sidebar:
    st.subheader(K.SIDEBAR_SUBTITLE(K.lang))
    if "retrived_text" in ss:
        st.markdown(ss["retrived_text"])




