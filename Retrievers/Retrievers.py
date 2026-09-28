from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
# from langchain_core.retrievers. import 
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = UnstructuredMarkdownLoader(file_path="./Retrievers/SampleDocument.md")
document = loader.load()

# print(document)

splitter = CharacterTextSplitter(
    separator=' ',
    chunk_size=50,          
    chunk_overlap=10,        
    length_function=len,    
    is_separator_regex=False 
)

chunks = splitter.split_documents(documents=document)

# print(chunks)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db = Chroma(
    embedding_function=embedding,
    persist_directory="kafka"
)

db.add_documents(chunks)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k":5
    }
)

query = "What is Kafka?"
results = retriever.invoke(query)

print(results)
