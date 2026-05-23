# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# load_dotenv()

# embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

# documents = [
#     "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
#     "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
#     "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
#     "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
#     "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
# ]

# query = 'tell me about MS Dhoni'

# doc_embedding = embedding.embed_documents(documents)
# query_embedding = embedding.embed_query(query)

# print(cosine_similarity([query_embedding],doc_embedding))





# using hugging face
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query1 = "tell me about MS Dhoni"
query2 = "tell me about batting in cricket"

doc_embedding = embedding.embed_documents(documents)
query_embedding1 = embedding.embed_query(query1)
query_embedding2 = embedding.embed_query(query2)


score1 = cosine_similarity([query_embedding1],doc_embedding)[0]
score2 = cosine_similarity([query_embedding2],doc_embedding)[0]

index, score = sorted(list(enumerate(score1)),key=lambda x:x[1])[-1]

print(query1)
print(documents[index])
print("Similarity Score is: ",score)

