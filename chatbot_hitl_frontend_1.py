import streamlit as st
from chatbot_hitl_backend_1 import chatbot
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt, Command
import uuid
import os
import tempfile

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "waiting_for_hitl" not in st.session_state:
    st.session_state.waiting_for_hitl = False

if "hitl_prompt" not in st.session_state:
    st.session_state.hitl_prompt = None
if "file" not in st.session_state:
    st.session_state.file=""

st.title("LangGraph Chatbot")


with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "xlsx", "xls", "tsv"])
    if os.path.isfile(st.session_state.file+".png"):
        with open(st.session_state.file+".png", "rb") as file:
            st.download_button(label="Download PNG Image",
            data=file,
            file_name="downloaded_image.png",
            mime="image/png")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

    # Create a temporary file that persists long enough to run the command
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name  # This is the physical path string
        #print(temp_path)
        st.session_state.file=temp_path

print(st.session_state.file)
# Normal chat input
if not st.session_state.waiting_for_hitl:
    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.write(user_input+st.session_state.file)

        state = {
            "messages": [HumanMessage(content=user_input)],
            "file":st.session_state.file
        }

        result = chatbot.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            },
        )

        interrupts = result.get("__interrupt__", [])

        if interrupts:
            st.session_state.waiting_for_hitl = True
            st.session_state.hitl_prompt = interrupts[0].value
            st.rerun()

        else:
            bot_response = result["messages"][-1].content

            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )

            with st.chat_message("assistant"):
                st.write(bot_response)
            #st.rerun()



# HITL approval section
if st.session_state.waiting_for_hitl:
    st.warning(st.session_state.hitl_prompt)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Approve"):
            result = chatbot.invoke(
                Command(resume="yes"),
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                },
            )

            bot_response = result["messages"][-1].content

            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )

            st.session_state.waiting_for_hitl = False
            st.session_state.hitl_prompt = None

            st.rerun()

    with col2:
        if st.button("Reject"):
            result = chatbot.invoke(
                Command(resume="no"),
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                },
            )

            bot_response = result["messages"][-1].content

            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )

            st.session_state.waiting_for_hitl = False
            st.session_state.hitl_prompt = None

            st.rerun()
