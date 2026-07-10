import streamlit as st
from chatbot_hitl_backend_1 import chatbot
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt, Command
import uuid
import os
import tempfile
import glob

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "waiting_for_hitl" not in st.session_state:
    st.session_state.waiting_for_hitl = False

if "hitl_prompt" not in st.session_state:
    st.session_state.hitl_prompt = None
if "files" not in st.session_state:
    st.session_state.files=[]

st.title("Bioinformatics Assistant")


with st.sidebar:
    uploaded_files = st.file_uploader("Choose a file", type=["xlsx", "xls"],accept_multiple_files=True)
    
if (uploaded_files!=[] or uploaded_files!=None) and st.session_state.files==[]:
    for each_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(each_file.name)[-1]) as temp_file:
            temp_file.write(each_file.getvalue())
            temp_path = temp_file.name  # This is the physical path string
            st.session_state.files.append(temp_path)

if (uploaded_files==[] or uploaded_files==None) and st.session_state.files!=[]:
    for remove_files in st.session_state.files:
        os.remove(remove_files)
    st.session_state.files=[]
with st.sidebar:
    if (uploaded_files!=[] or uploaded_files!=None) and st.session_state.files!=[]:
        if len(glob.glob(st.session_state.files[0]+"_*")) == 1:
            if os.path.isfile(glob.glob(st.session_state.files[0]+"_*")[0]):
                with open(glob.glob(st.session_state.files[0]+"_*")[0], "rb") as file:
                    st.download_button(label="Download Processed File",
                    data=file,
                    file_name="Downloaded_file."+glob.glob(st.session_state.files[0]+"_*")[0].split(".")[-1],
                    mime="application/octet-stream")




with st.sidebar:
    if (uploaded_files!=[] or uploaded_files!=None) and st.session_state.files!=[]:
        if len(glob.glob(st.session_state.files[0]+"_*")) == 2:
            if os.path.isfile(glob.glob(st.session_state.files[0]+"_*")[0]):
                with open(glob.glob(st.session_state.files[0]+"_*")[0], "rb") as file:
                    st.download_button(label="Download Processed File",
                    data=file,
                    file_name="Downloaded_file."+glob.glob(st.session_state.files[0]+"_*")[0].split(".")[-1],
                    mime="application/octet-stream")

            if os.path.isfile(glob.glob(st.session_state.files[0]+"_*")[1]):
                with open(glob.glob(st.session_state.files[0]+"_*")[1], "rb") as file:
                    st.download_button(label="Download Processed File",
                    data=file,
                    file_name="Downloaded_file."+glob.glob(st.session_state.files[0]+"_*")[1].split(".")[-1],
                    mime="application/octet-stream")


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Normal chat input
if not st.session_state.waiting_for_hitl:
    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.write(user_input)

        state = {
            "messages": [HumanMessage(content=user_input)],
            "files":st.session_state.files
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
            st.rerun()



# HITL approval section
if st.session_state.waiting_for_hitl:
    st.warning(st.session_state.hitl_prompt)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Human"):
            result = chatbot.invoke(
                Command(resume="human"),
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
        if st.button("Mouse"):
            result = chatbot.invoke(
                Command(resume="mouse"),
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
