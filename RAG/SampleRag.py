# ### Prepare
# 1. Load documents using a document loader.
# 2. Split documents into chunks using a text splitter.
# 3. Generate embeddings for the chunks.
# 4. Store the chunks and their embeddings in a vector store.

# ### Utilize
# 1. Receive the user query.
# 2. Generate an embedding for the user query.
# 3. Retrieve relevant documents/chunks from the vector store.
# 4. Pass the system prompt, user query, and retrieved context to the LLM.
# 5. Parse the LLM output.


from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import MarkdownTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()


# --------------------------------------------------
# 1. Load document
# --------------------------------------------------

document_loader = UnstructuredMarkdownLoader(
    "./RAG/SampleDocument.md"
)

documents = document_loader.load()


# --------------------------------------------------
# 2. Split document into chunks
# --------------------------------------------------

text_splitter = MarkdownTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
)

chunks = text_splitter.split_documents(documents)


# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# --------------------------------------------------
# 4. Store chunks + embeddings in Chroma
# --------------------------------------------------

database = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="samplerag"
)


# --------------------------------------------------
# 5. Create retriever
# --------------------------------------------------

retriever = database.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.2
    }
)


# --------------------------------------------------
# 6. LLM
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.1
)


# --------------------------------------------------
# 7. Prompt
# --------------------------------------------------

prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks.

Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

# Context:
{context}

# Question:
{question}

# Answer:"""
)


# --------------------------------------------------
# 8. Convert Documents to text
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        document.page_content
        for document in docs
    )


# --------------------------------------------------
# 9. RAG chain
# --------------------------------------------------

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)


# # --------------------------------------------------
# # 10. Query
# # --------------------------------------------------

# question = "Upon whose name the Kafka software is named?"

# response = chain.invoke(question)

# print(response)

# --------------------------------------------------
# 11. Memory
# --------------------------------------------------

store = {}
