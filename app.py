import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import streamlit as st
import google.generativeai as genai

# 경로 설정
data_path = './data'
module_path = './modules'

# Gemini 모델 설정
genai.configure(api_key="AIzaSyADPxt3kzIIk9SA03BG4LoBi-Kcq5AhXqw")  # 실제 API 키 사용
model = genai.GenerativeModel("gemini-1.5-flash")

# 데이터 로드
df = pd.read_csv(os.path.join(data_path, "JEJU_DATA.csv"), encoding='cp949')
df_tour = pd.read_csv(os.path.join(data_path, "JEJU_TOUR.csv"), encoding='cp949')
text_tour = df_tour['text'].tolist()

# 최신연월 데이터만 사용
df = df[df['기준연월'] == df['기준연월'].max()].reset_index(drop=True)

# Streamlit 설정
st.set_page_config(page_title="🍊참신한 제주 맛집!")
st.title("혼저 옵서예!👋")
st.subheader("군맛난 제주 밥집🧑‍🍳 추천해드릴게예")
st.write("#흑돼지 #갈치조림 #옥돔구이 #고사리해장국 #전복뚝배기 #한치물회 #빙떡 #오메기떡..🤤")

# 이미지 추가
image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTHBMuNn2EZw3PzOHnLjDg_psyp-egZXcclWbiASta57PBiKwzpW5itBNms9VFU8UwEMQ&usqp=CAU"
st.image(image_path, width=500)

# 대화 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}]

# 메시지 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 챗 기록 초기화 버튼
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face 임베딩 모델 및 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return index
    else:
        raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {index_path}")

# 텍스트 임베딩 생성
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# 텍스트 임베딩 로드
embeddings = np.load(os.path.join(module_path, 'embeddings_array_file.npy'))

# 관광지 텍스트 임베딩 생성
def get_huggingface_embeddings(texts):
    inputs_tour = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs_tour)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# 관광지 임베딩 로드 및 FAISS 인덱스 생성
embeddings_tour = get_huggingface_embeddings(text_tour)
index_tour = faiss.IndexFlatL2(embeddings_tour.shape[1])
index_tour.add(embeddings_tour)

# FAISS를 활용한 응답 생성
def generate_response_with_faiss(question, df, embeddings, model, df_tour, embeddings_tour, k=3):
    index = load_faiss_index()
    query_embedding = embed_text(question).reshape(1, -1)
    distances, indices = index.search(query_embedding, k * 3)
    
    query_embedding_tour = get_huggingface_embeddings([question]).reshape(1, -1)
    distances_tour, indices_tour = index_tour.search(query_embedding_tour, k)

    filtered_df = df.iloc[indices[0, :]].reset_index(drop=True)
    filtered_df_tour = df_tour.iloc[indices_tour[0, :]].reset_index(drop=True)

    if filtered_df.empty:
        return "질문과 일치하는 가게가 없습니다."

    reference_info = "\n".join(filtered_df['text'])
    reference_tour = "\n".join(filtered_df_tour['text'])

    prompt = f"""
    질문: {question}
    대답시 필요한 내용: 근처 음식점을 추천할때는 위도와 경도를 비교해서 가까운 곳으로 추천해줘야해. \n위도와 경도의 정확도가 99.99%내외가 아니라면 근처가 아니라고 꼭 알려주고, 차로 얼마나 걸릴지 알려줘. 대답할때 위도, 경도는 안 알려줘도 돼.\n대답해줄때 업종별로 가능하면 하나씩 추천해줘. 그리고 추가적으로 그 중에서 가맹점개점일자가 오래되고 이용건수가 많은 음식점(오래된맛집)과 가맹점개점일자가 최근이고 이용건수가 많은 음식점(새로운맛집)을 각각 추천해줬으면 좋겠어.
    참고할 정보: {reference_info}
    참고할 관광지 정보: {reference_tour}
    응답:"""

    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else response

# 사용자 입력 처리 및 응답 생성
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# 사용자가 입력한 질문에 대한 응답 생성
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = generate_response_with_faiss(prompt, df, embeddings, model, df_tour, embeddings_tour)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
