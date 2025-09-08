from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Seoul, the capital of South Korea, is a huge metropolis where modern skyscrapers, high-tech subways and pop culture meet Buddhist temples, palaces and street markets. Notable attractions include futuristic Dongdaemun Design Plaza, a convention hall with curving architecture and a rooftop park",
    "Jeju Island (or Jejudo) is South Korea's largest island, a popular tourist destination known for its volcanic landscape, beautiful beaches, and lush natural beauty, often called the Hawaii of Korea. A UNESCO World Heritage Site, the island features the dormant volcano Hallasan, coastal trails, lava tubes like Manjanggul Cave, and scenic waterfalls.",
    "Busan, a large port city in South Korea, is known for its beaches, mountains and temples. Busy Haeundae Beach features the Sea Life Aquarium, plus a Folk Square with traditional games such as tug-of-war, while Gwangalli Beach has many bars and views of modern Diamond Bridge. Beomeosa Temple, a Buddhist shrine founded in 678 A.D",
    "South Gyeongsang is a province of South Korea that extends along the Korea Strait, west of Busan city. The provincial capital, Changwon, is famed for the light shows of Yongji Lake and for cherry-blossom viewing. Northeast of Changwon, the Junam Wetlands Park hosts migratory birds. Farther north, the Upo Wetland is known for summer fireflies.",
]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

query = "tell me about the Jeju Island"

doc_embeddings = embedding.embed_documents(documents)

query_embeddings = embedding.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:",score)