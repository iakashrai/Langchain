from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_community.embeddings import e

loader = UnstructuredMarkdownLoader(file_path="./TextSplitter/SampleDocument.md")
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