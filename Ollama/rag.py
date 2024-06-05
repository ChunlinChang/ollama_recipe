from langchain.vectorstores import Chroma  # 導入 Chroma 向量存儲庫，用於文檔向量化
from langchain.chat_models import ChatOllama  # 導入 ChatOllama 模型，用於回答問題
from langchain.embeddings import FastEmbedEmbeddings  # 導入 FastEmbedEmbeddings，用於文本向量化
from langchain.schema.output_parser import StrOutputParser  # 導入 StrOutputParser，用於解析模型輸出結果
from langchain.document_loaders import PyPDFLoader  # 導入 PyPDFLoader，用於從 PDF 文件加載文檔
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 導入 RecursiveCharacterTextSplitter，用於將文本拆分為小塊進行處理
from langchain.schema.runnable import RunnablePassthrough  # 導入 RunnablePassthrough，用於將問題通過管道傳遞給模型
from langchain.prompts import PromptTemplate  # 導入 PromptTemplate，用於提示用戶輸入問題
from langchain.vectorstores.utils import filter_complex_metadata  # 導入 filter_complex_metadata，用於過濾文檔的複雜元數據
from langchain_community.document_loaders.csv_loader import CSVLoader  # 導入 CSVLoader，用於從 CSV 文件加載文檔

class ChatPDF:
    vector_store = None  # 文檔向量存儲庫，初始為 None
    retriever = None  # 文本檢索器，初始為 None
    chain = None  # 問答流程，初始為 None

    def __init__(self):
        self.model = ChatOllama(model="llama3")  # 初始化 ChatOllama 模型
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)  # 初始化 RecursiveCharacterTextSplitter，用於文本拆分
        self.prompt = PromptTemplate.from_template(  # 初始化提示模板，用於提示用戶輸入問題
            """
            <s> [INST]
            
            你是一位廚師，你需要解決使用者料理部份的問題，請用正體中文以客觀角度回覆問題。
            若使用者詢問食譜、製作方法須明確列出準備食材與料理步驟。
            僅使用以下檢索到的內容來為使用者構建答案。如果您不知道答案，只需說您不知道。
            
            [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()  # 從 PDF 文件加載文檔
        chunks = self.text_splitter.split_documents(docs)  # 將文檔拆分為小塊進行處理
        chunks = filter_complex_metadata(chunks)  # 過濾文檔的複雜元數據

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())  # 將文檔向量化
        self.retriever = vector_store.as_retriever(  # 將向量存儲庫轉換為文本檢索器
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        # 定義問答流程
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                    | self.prompt
                    | self.model
                    | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)  # 啟動問答流程，對問題進行回答

    def clear(self):
        self.vector_store = None  # 清除文檔向量存儲庫
        self.retriever = None  # 清除文本檢索器
        self.chain = None  # 清除問答流程
