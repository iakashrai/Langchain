from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader

# loader = PyPDFLoader("./DocumentLoader/CommonLeetcodePatterns.pdf")

# document = loader.load()

# print(document)


loader = WebBaseLoader(web_path="https://google.com")

document = loader.load()

print(document)