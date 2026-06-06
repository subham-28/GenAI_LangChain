from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load() # if number of documents less
# docs = loader.lazy_load() # if number of document more - loads documents as need not all at a time


print(docs)