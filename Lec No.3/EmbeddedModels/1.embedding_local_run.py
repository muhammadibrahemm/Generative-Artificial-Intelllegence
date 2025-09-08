from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "South Korea is the Best Country in the World"

vector = embedding.embed_query(text)

print(str(vector))