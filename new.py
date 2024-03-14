import pandas as pd
import json
import os

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.document_loaders import WebBaseLoader

# 대화 기록 파일 경로
conversation_history_path = "conversation_history.json"

# 대화 기록 로드
conversation_history = {}
if os.path.exists(conversation_history_path):
    with open(conversation_history_path, "r") as file:
        conversation_history = json.load(file)

# 데이터프레임 로드
def get_faq_file(file_path):

    pkl_data = pd.read_pickle(file_path)

    ques = pd.DataFrame.from_dict([pkl_data.keys()])
    ques = pd.DataFrame.transpose(ques)
    ques = ques.rename(columns={0:"ques"})
    ans = pd.DataFrame.from_dict([pkl_data.values()])
    ans = pd.DataFrame.transpose(ans)
    ans = ans.rename(columns={0:"ans"})

    pkl_data = pd.concat([ques, ans], axis=1)
    return pkl_data

# 텍스트 분할
def get_text_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(text)
    return texts

# 벡터 스토어 
def get_vector_store(chunks):
    persist_directory = 'db'

    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=chunks,
                                    embedding=embedding,
                                    persist_directory=persist_directory
                                    )
    # DB 상태 보존
    vectordb.persist() 
    return vectordb

# 대화 모델 선정 
def get_chain(chunks):
    llm = OpenAI()
    vectordb = get_vector_store(chunks)
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vectordb.as_retriever(search_kwargs={'k':1}),
                                        return_source_documents=True)
    return qa_chain

def get_llm_response(llm_response):
    print(f"Chatbot: {llm_response['result']}")
    print(f"(Details): {source}")
    for source in llm_response['source_documents'][0].metadata['ans'].split('.')[:2]:
        print(source)

def save_conversation_history(conversation_history):
    with open(conversation_history_path, "w") as file:
        json.dump(conversation_history, file)

def main():
    datas = get_faq_file("./final_result.pkl")
    loader = DataFrameLoader(datas, page_content_column="ques")
    #e_path = "https://help.sell.smartstore.naver.com/index.help"
    #loader = WebBaseLoader(e_path)
    documents = loader.load()
    chunks = get_text_chunk(documents)
    vectordb = get_vector_store(chunks)

    qa_chain = get_chain(chunks)

    flag = 0
    while True:
        # 사용자 입력 받기
        user_input = input("You: ")
        
        # 대화 기록에 사용자 입력 추가
        conversation_history["user_input"] = user_input 
        save_conversation_history(conversation_history)
            
        # 챗봇 응답 출력
        if flag == 0:
            llm_response = qa_chain(user_input)
            get_llm_response(llm_response)
            flag += 1
        else:
            user_input = user_input + conversation_history["bot_response"]

            # 이전 대화 맥락에서 답을 찾도록 함 
            qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                        chain_type="stuff",
                                        retriever=vectordb,
                                        return_source_documents=True)
            llm_response = qa_chain(user_input)
            get_llm_response(llm_response)
        conversation_history["bot_response"] = llm_response["source_documents"][0].metadata["ans"]
        save_conversation_history(conversation_history)


if __name__ == "__main__":
    main()