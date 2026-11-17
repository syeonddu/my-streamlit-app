
import os
import hashlib
import tempfile
from pathlib import Path
from operator import itemgetter

import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ==========================================
# 1. 문서 포맷팅
# ==========================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ==========================================
# 2. 프롬프트
# ==========================================

def get_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """당신은 AIVLE School DX 트랙 학습 도우미입니다.

아래 [참고 문서]만 근거로 질문에 답하세요.

답변 규칙:
1. 질문에 직접 답하고 필요한 경우 항목별로 설명하세요.
2. 공식 안내, 개인 후기, 작성자의 학습 제안을 구분하세요.
3. 문서에 근거가 없으면
   "백서에서 해당 내용을 확인할 수 없습니다."라고 말하세요.
4. 지원 조건, 일정, 수료 기준처럼 변경될 수 있는 정보는
   해당 기수의 공식 공지 확인이 필요함을 안내하세요.
5. 문서에 없는 사실이나 수치를 만들지 마세요.
6. 친절하고 이해하기 쉬운 한국어로 답하세요.

[참고 문서]
{context}"""
        ),
        ("human", "{input}")
    ])


# ==========================================
# 3. RAG 체인 생성
# ==========================================

def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | get_prompt()
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ==========================================
# 4. 기존 ChromaDB 불러오기
# ==========================================

@st.cache_resource(show_spinner=False)
def get_default_chain():

    chroma_dir = Path(__file__).resolve().parent / "chroma_db"

    if not chroma_dir.is_dir():
        return None

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        collection_name="aivle_docs",
        embedding_function=embeddings,
        persist_directory=str(chroma_dir)
    )

    if vectorstore._collection.count() == 0:
        return None

    return build_chain(vectorstore)


# ==========================================
# 5. 새 PDF로 RAG 파이프라인 생성
# ==========================================

def rag_pipeline(filepath):

    # PDF 읽기
    loader = PyMuPDF4LLMLoader(str(filepath))
    docs = loader.load()

    docs = [
        doc for doc in docs
        if doc.page_content.strip()
    ]

    if not docs:
        raise ValueError("PDF에서 읽을 수 있는 내용이 없습니다.")

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    # 임베딩
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 업로드한 PDF 전용 임시 벡터 저장소
    vectorstore = Chroma(
        collection_name="uploaded_aivle_docs",
        embedding_function=embeddings
    )

    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    return build_chain(vectorstore)


# ==========================================
# 6. Streamlit 화면
# ==========================================

st.title("AIVLE School 학습 도우미")


# API 키 확인
try:
    secret_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    secret_key = None

if secret_key:
    os.environ["OPENAI_API_KEY"] = secret_key

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY가 없습니다.")
    st.stop()


# 세션 상태 초기화
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "active_source" not in st.session_state:
    st.session_state.active_source = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 7. PDF 업로드
# ==========================================

uploaded_pdf = st.file_uploader(
    "AIVLE School 백서 PDF 업로드",
    type=["pdf"]
)

pdf_bytes = (
    uploaded_pdf.getvalue()
    if uploaded_pdf is not None
    else None
)

# 현재 사용하는 문서 구분
source_id = (
    "default"
    if pdf_bytes is None
    else "upload:" + hashlib.sha256(pdf_bytes).hexdigest()
)

# 문서가 바뀌면 대화 초기화
if st.session_state.active_source != source_id:
    st.session_state.active_source = source_id
    st.session_state.rag_chain = None
    st.session_state.messages = []


# ==========================================
# 8. 사용할 RAG 체인 준비
# ==========================================

if pdf_bytes is not None:

    # 새로운 PDF가 업로드된 경우
    if st.session_state.rag_chain is None:

        temp_path = None

        try:
            with st.spinner("PDF를 분석하고 있습니다..."):

                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False
                ) as temp_pdf:

                    temp_pdf.write(pdf_bytes)
                    temp_path = temp_pdf.name

                st.session_state.rag_chain = rag_pipeline(temp_path)

            st.success("PDF 분석 완료! 질문해 주세요.")

        except Exception as e:
            st.error(f"PDF 분석 중 오류: {e}")

        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

else:

    # PDF 업로드가 없으면 기존 chroma_db 사용
    if st.session_state.rag_chain is None:

        try:
            st.session_state.rag_chain = get_default_chain()

        except Exception as e:
            st.warning(f"벡터 저장소 불러오기 오류: {e}")

    if st.session_state.rag_chain is None:
        st.info(
            "저장된 백서 데이터가 없습니다. "
            "위에서 PDF를 업로드하면 질문할 수 있습니다."
        )


# ==========================================
# 9. 기존 채팅 기록 표시
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# ==========================================
# 10. 채팅 입력창
# ==========================================

question = st.chat_input(
    "백서에 대해 질문해 주세요.",
    disabled=st.session_state.rag_chain is None
)


# ==========================================
# 11. 질문 및 답변 처리
# ==========================================

if question and st.session_state.rag_chain is not None:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        try:
            with st.spinner("답변을 생성하고 있습니다..."):

                answer = st.session_state.rag_chain.invoke({
                    "input": question
                })

            st.write(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"답변 생성 중 오류: {e}")
