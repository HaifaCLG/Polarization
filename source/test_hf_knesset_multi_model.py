from sentence_transformers import SentenceTransformer
sentences = ["זה משפט ראשון לדוגמה", "זה המשפט השני"]

model = SentenceTransformer('GiliGold/Knesset-multi-e5-large')
embeddings = model.encode(sentences)
print(embeddings)
