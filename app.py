import pymupdf
import faiss                                                   
pdf=pymupdf.open("ragpdf.pdf")                      # LOADS THE WHOLE PDF
text=""                
for pages in pdf:                                   # FOR EXERY PAGE IN THE PDF EXTRACT THE TEXTS AND PRINT
    text=text+ pages.get_text()                            
   
chunks=[]
for i in range(0,len(text),500):
    chunk=text[i:i+500]
    chunks.append(chunk)  
        
from sentence_transformers import SentenceTransformer
model= SentenceTransformer("all-MiniLM-L6-v2")                       
embeddings= model.encode(chunks)

dimension=len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

question=input("Ask a Question: ")
question_embedding=model.encode([question])
distances,indices=index.search(question_embedding,3)

context=""
for i in indices[0]:
    context=context+chunks[i]

prompt = "Use the following context to answer the question:\n" + context + "\nQuestion: " + question
# print(prompt)
from google import genai
client=genai.Client()
response = client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
)

print(response.output_text)
