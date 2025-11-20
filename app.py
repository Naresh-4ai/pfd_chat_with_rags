import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import hashlib

# Load environment variables
load_dotenv()

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI()

@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-large")

def get_pdf_hash(pdf_file):
    """Generate a unique hash for the PDF file"""
    return hashlib.md5(pdf_file.getvalue()).hexdigest()

def process_pdf(pdf_file, collection_name):
    """Process PDF and store in vector database"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Loading
        with st.spinner("📄 Loading PDF..."):
            loader = PyPDFLoader(file_path=tmp_path)
            docs = loader.load()
        
        # Chunking
        with st.spinner("✂️ Splitting document into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=400
            )
            split_docs = text_splitter.split_documents(documents=docs)
        
        # Vector Embeddings
        with st.spinner("🔢 Creating embeddings and storing in database..."):
            embedding_model = get_embedding_model()
            
            # Store in Qdrant
            vector_store = QdrantVectorStore.from_documents(
                documents=split_docs,
                url="http://localhost:6333",
                collection_name=collection_name,
                embedding=embedding_model,
                force_recreate=True
            )
        
        return True, len(split_docs)
    
    except Exception as e:
        return False, str(e)
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def get_answer(query, collection_name):
    """Get answer from vector database"""
    try:
        embedding_model = get_embedding_model()
        
        # Connect to existing collection
        vector_db = QdrantVectorStore.from_existing_collection(
            url="http://localhost:6333",
            collection_name=collection_name,
            embedding=embedding_model
        )
        
        # Vector Similarity Search
        search_results = vector_db.similarity_search(
            query=query,
            k=4  # Return top 4 relevant chunks
        )
        
        if not search_results:
            return "I couldn't find relevant information in the PDF to answer your question."
        
        # Build context from search results
        context = "\n\n---\n\n".join([
            f"Content: {result.page_content}\n"
            f"Page: {result.metadata.get('page', 'N/A')}\n"
            f"Source: {result.metadata.get('source', 'N/A')}"
            for result in search_results
        ])
        
        # Create system prompt
        SYSTEM_PROMPT = f"""You are a helpful AI Assistant that answers questions based on the context 
retrieved from a PDF file. 

Instructions:
- Answer the user's question based ONLY on the provided context
- If the answer is in the context, provide a clear and concise response
- Always mention the page number(s) where the information was found
- If the question cannot be answered from the context, politely say so
- Be precise and informative

Context:
{context}
"""
        
        # Get completion from OpenAI
        client = get_openai_client()
        chat_completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"Error getting answer: {str(e)}"

def main():
    st.set_page_config(
        page_title="PDF Chat Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Chat Assistant")
    st.markdown("Upload a PDF and ask questions about its content!")
    
    # Initialize session state
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = None
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📤 Upload PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF file to chat with"
        )
        
        if uploaded_file is not None:
            pdf_hash = get_pdf_hash(uploaded_file)
            current_collection = f"pdf_collection_{pdf_hash}"
            
            # Check if this is a new PDF
            if st.session_state.collection_name != current_collection:
                st.session_state.pdf_processed = False
                st.session_state.collection_name = current_collection
                st.session_state.chat_history = []
                st.session_state.pdf_name = uploaded_file.name
            
            # Display PDF info
            st.info(f"**File:** {uploaded_file.name}\n\n**Size:** {uploaded_file.size / 1024:.2f} KB")
            
            # Process PDF button
            if not st.session_state.pdf_processed:
                if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                    success, result = process_pdf(uploaded_file, current_collection)
                    
                    if success:
                        st.session_state.pdf_processed = True
                        st.success(f"✅ PDF processed successfully!\n\n{result} chunks created.")
                        st.rerun()
                    else:
                        st.error(f"❌ Error processing PDF: {result}")
            else:
                st.success("✅ PDF is ready for questions!")
                
                if st.button("🔄 Upload New PDF", use_container_width=True):
                    st.session_state.pdf_processed = False
                    st.session_state.collection_name = None
                    st.session_state.chat_history = []
                    st.session_state.pdf_name = None
                    st.rerun()
        
        st.divider()
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Upload a PDF file first
        - Click 'Process PDF' to index it
        - Ask questions about the content
        - Get answers with page references
        """)
    
    # Main chat interface
    if st.session_state.pdf_processed:
        st.success(f"💬 Chatting with: **{st.session_state.pdf_name}**")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = get_answer(prompt, st.session_state.collection_name)
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    else:
        st.info("👈 Please upload a PDF file from the sidebar to get started!")
        
        # Show example usage
        st.markdown("### 📖 How to use:")
        st.markdown("""
        1. **Upload** a PDF file using the sidebar
        2. **Click** 'Process PDF' to index the document
        3. **Ask** questions about the PDF content
        4. **Get** answers with relevant page references
        """)
        
        st.markdown("### ✨ Example Questions:")
        st.code("""
• "What is this document about?"
• "Summarize the main points"
• "What does it say about [specific topic]?"
• "Find information about [keyword]"
        """)

if __name__ == "__main__":
    main()