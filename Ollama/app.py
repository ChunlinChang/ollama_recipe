import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

# 設置 Streamlit 頁面配置
st.set_page_config(page_title="ChatPDF")

# 顯示聊天消息
def display_messages():
    st.subheader("Chat") #子標題
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()  # 用於顯示思考過程的旋轉圖標

# 處理用戶輸入
def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)  # 呼叫聊天機器人進行回答

        st.session_state["messages"].append((user_text, True))  # 添加用戶消息
        st.session_state["messages"].append((agent_text, False))  # 添加機器人消息

# 讀取並保存上傳的文件
def read_and_save_file():
    st.session_state["assistant"].clear()  # 清除先前的聊天記錄
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    print("runed read_and_save_file(...)")

    predefined_files = ["Ollama/food_1.pdf"] #預設PDF
    
    for file_path in predefined_files:
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file_path}"):
            st.session_state["assistant"].ingest(file_path)  # 將文件內容導入聊天機器人
    
    # for file in st.session_state["file_uploader"]:
    #     with tempfile.NamedTemporaryFile(delete=False) as tf:
    #         tf.write(file.getbuffer())  # 寫入臨時文件
    #         file_path = tf.name

    #     with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
    #         st.session_state["assistant"].ingest(file_path)  # 將文件內容導入聊天機器人
    #     os.remove(file_path)  # 刪除臨時文件

# 主頁面函數
def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []  # 初始化消息列表
        st.session_state["assistant"] = ChatPDF()  # 初始化聊天機器人
        st.session_state["ingestion_spinner"] = st.empty()  # 初始化文件導入過程的旋轉圖標

        # 初始化时读取和处理预定义的PDF文件
        read_and_save_file()
        
    st.header("ChatPDF") #大標題

    #st.subheader("Upload a document")
    # st.file_uploader(
    #     "Upload document",
    #     type=["pdf"],
    #     key="file_uploader",
    #     on_change=read_and_save_file,  # 文件上傳後執行的回調函數
    #     label_visibility="collapsed",
    #     accept_multiple_files=True,  # 允許多文件上傳
    # )

    st.session_state["ingestion_spinner"] = st.empty()  # 用於顯示文件導入過程的旋轉圖標

    display_messages()  # 顯示聊天消息
    st.text_input("Message", key="user_input", on_change=process_input)  # 用戶輸入框

# 主程序入口
if __name__ == "__main__":
    page()
