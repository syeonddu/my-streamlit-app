# AIVLE School 학습 도우미: 교안 미션 3 기본 기능
# 기존 확정 RAG 파이프라인은 아래에 원문 그대로 유지했습니다.
# Google Colab에서 Streamlit 서버로 실행합니다.

import os
import hashlib
import tempfile
from pathlib import Path
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from operator import itemgetter


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def rag_pipeline(filepath):
    # 1. PDF 파일 읽기
    loader = PyMuPDF4LLMLoader(str(filepath))
    docs = loader.load()

    # 비어 있는 문서 제외
    docs = [
        doc for doc in docs
        if doc.page_content.strip()
    ]

    if not docs:
        raise ValueError("PDF에서 읽을 수 있는 본문이 없습니다.")

    print(f"1. PDF 읽기 완료: {len(docs)}개 Document")

    # 2. 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print(f"2. 문서 분할 완료: {len(chunks)}개 조각")

    # 3. 임베딩 모델
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 4. 벡터 저장소 구성
    vectorstore = Chroma(
        collection_name="aivle_docs",
        embedding_function=embeddings
    )

    # 실습 재실행 시 같은 컬렉션에 문서가 중복 추가되지 않게 초기화
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    print("3. 임베딩 및 벡터 저장 완료")

    # 5. 검색기 구성
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # 6. 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """당신은 AIVLE School DX 트랙 학습 도우미입니다.

아래 [참고 문서]만 근거로 질문에 답하세요.

답변 규칙:
1. 질문에 직접 답하고, 필요한 경우 항목별로 설명하세요.
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

    # 7. 답변 모델
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # 8. RAG 체인 연결
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("4. RAG 체인 구성 완료")

    return rag_chain


# ===== Streamlit 기본 화면: 기능 연결만 구현, 별도 디자인 없음 =====
st.title("AIVLE School 학습 도우미")

# 코랩 실행 셀에서 OPENAI_API_KEY 환경변수를 설정합니다.
try:
    secret_api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    secret_api_key = None

if secret_api_key:
    os.environ["OPENAI_API_KEY"] = secret_api_key

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Secrets 또는 환경변수를 확인해 주세요.")
    st.stop()

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_pdf = st.file_uploader("AIVLE School 백서 PDF 업로드", type=["pdf"])

if uploaded_pdf is None:
    # 업로드를 취소했다면 이전 문서의 검색 결과는 사용하지 않습니다.
    st.session_state.rag_chain = None
    st.session_state.pdf_hash = None
    st.session_state.messages = []
    st.info("백서 PDF를 업로드하면 자동으로 문서를 처리하고 질문할 수 있습니다.")
else:
    pdf_bytes = uploaded_pdf.getvalue()
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # 동일 PDF는 매번 재벡터화하지 않습니다.
    # 다른 PDF를 업로드할 때만 기존 체인을 비우고 새 체인을 만듭니다.
    if current_hash != st.session_state.pdf_hash:
        st.session_state.rag_chain = None
        st.session_state.pdf_hash = None
        st.session_state.messages = []

        if not pdf_bytes:
            st.error("비어 있는 PDF 파일입니다. 다른 파일을 선택해 주세요.")
            st.stop()

        temp_path = None
        try:
            with st.spinner("PDF를 읽고 RAG 파이프라인을 구성하고 있습니다..."):
                # 확정 함수는 파일 객체가 아니라 PDF '경로'를 받습니다.
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    temp_path = tmp.name
                chain = rag_pipeline(temp_path)
                st.session_state.rag_chain = chain
                st.session_state.pdf_hash = current_hash
            st.success("PDF 분석 완료! 질문을 입력해 주세요.")
        except Exception as exc:
            st.error(f"PDF 처리 중 오류가 발생했습니다: {exc}")
            st.stop()
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)

    # Streamlit은 입력마다 전체 스크립트를 재실행하므로 메시지 기록을 상태로 유지합니다.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input(
        "백서에 대해 질문해 주세요.",
        disabled=st.session_state.rag_chain is None,
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("답변을 생성하고 있습니다..."):
                    answer = st.session_state.rag_chain.invoke({"input": question})
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error(f"답변 생성 중 오류가 발생했습니다: {exc}")
