import streamlit as st
from rag_app import RAGPipeline

st.set_page_config(layout="wide")
st.title("RAG QA System")

@st.cache_resource
def get_rag_pipeline():
    rag = RAGPipeline(storage_dir="streamlit_rag_storage")
    rag.load()
    return rag

rag = get_rag_pipeline()
kb_loaded = (rag.vector_db is not None and len(rag.vector_db.documents) > 0)

if not kb_loaded:
    st.warning("Knowledge base is not loaded. Please ensure it exists in storage.")
    st.stop()

st.header("Ask a Question")
question = st.text_input("Enter your question here")

k = st.slider("Number of documents to retrieve (k)", min_value=1, max_value=10, value=3)

if st.button("Get Answer"):
    if not question.strip():
        st.error("Please enter a question to query.")
    else:
        with st.spinner("Retrieving answer..."):
            result = rag.query(question, k=k)
            st.subheader("Answer")
            st.write(result["response"])

            st.subheader(f"Retrieved Documents ({result['num_retrieved']})")
            for i, doc in enumerate(result["retrieved_documents"], 1):
                st.markdown(f"**{i}. {doc['title']}** (score: {doc['score']:.3f})")
                st.text_area(f"Document {i} Content", value=doc["content"], height=200, key=f"doc_content_{i}")
                st.write(f"[Source]({doc['url']})")
