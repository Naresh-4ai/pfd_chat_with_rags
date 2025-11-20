# 📚 PDF Chat with RAGs

An intelligent PDF chat application built with Streamlit, LangChain, and Qdrant vector database. Upload any PDF and have natural conversations about its content using RAG (Retrieval Augmented Generation).

## 🌟 Features

- 📤 **Upload Any PDF** - Support for PDFs up to any size
- 💬 **Natural Chat Interface** - Conversational Q&A about your documents
- 🔍 **Semantic Search** - Vector-based similarity search using Qdrant
- 📖 **Page References** - Get exact page numbers for answers
- 🐳 **Docker Ready** - Fully containerized with Docker Compose
- 🎨 **Beautiful UI** - Clean and intuitive Streamlit interface
- ⚡ **Fast Responses** - Instant answers after initial processing

## 🏗️ Architecture

```
User Upload PDF → LangChain Processing → Text Chunking → 
OpenAI Embeddings → Qdrant Vector Store → 
Semantic Search → GPT-4 Response → User
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose installed
- OpenAI API Key
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Naresh-4ai/pfd_chat_with_rags.git
cd pfd_chat_with_rags
```

2. **Create `.env` file**
```bash
echo OPENAI_API_KEY=your-openai-api-key-here > .env
```

3. **Build and run with Docker**
```bash
docker-compose up --build
```

4. **Access the application**
Open your browser and navigate to: `http://localhost:8501`

## 📦 Project Structure

```
pfd_chat_with_rags/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── .env                  # Environment variables (create this)
├── .env.example          # Example environment file
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o
- **Embeddings**: OpenAI text-embedding-3-large
- **Vector Database**: Qdrant
- **Framework**: LangChain
- **Containerization**: Docker & Docker Compose

## 💻 Usage

1. **Upload PDF**: Click the upload button in the sidebar
2. **Process**: Click "🚀 Process PDF" to index the document
3. **Chat**: Ask questions about your PDF in natural language
4. **Get Answers**: Receive accurate answers with page references

### Example Questions

- "What is this document about?"
- "Summarize the main points on page 5"
- "What does it say about [specific topic]?"
- "Find all mentions of [keyword]"

## ⚙️ Configuration

### Chunk Settings (in `app.py`)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=400     # Overlap between chunks
)
```

### Model Configuration

```python
# Embedding Model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# LLM Model
model="gpt-4o"
```

## 🐳 Docker Commands

```bash
# Build and start services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f streamlit-app

# Rebuild specific service
docker-compose up --build streamlit-app
```

## 📊 Performance

### Processing Time (approximate)
- **Small PDF** (10-50 pages): 30-60 seconds
- **Medium PDF** (50-150 pages): 1-2 minutes
- **Large PDF** (150-300 pages): 3-5 minutes
- **Very Large PDF** (300+ pages): 5-10 minutes

### Query Response Time
- After processing: **2-3 seconds** per query

## 🔒 Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=http://qdrant:6333  # Docker service name
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🐛 Troubleshooting

### Common Issues

**1. Docker connection errors**
```bash
# Restart Docker services
docker-compose down
docker-compose up --build
```

**2. OpenAI API errors**
- Verify your API key in `.env`
- Check API quota/limits

**3. Memory issues with large PDFs**
- Increase Docker memory allocation
- Reduce chunk_size in configuration

## 📧 Contact

- **Author**: Naresh
- **GitHub**: [@Naresh-4ai](https://github.com/Naresh-4ai)
- **Project Link**: [https://github.com/Naresh-4ai/pfd_chat_with_rags](https://github.com/Naresh-4ai/pfd_chat_with_rags)

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM applications
- [Qdrant](https://qdrant.tech/) - Vector database
- [Streamlit](https://streamlit.io/) - Web framework
- [OpenAI](https://openai.com/) - LLM and embeddings

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ using LangChain, Streamlit, and Qdrant**
