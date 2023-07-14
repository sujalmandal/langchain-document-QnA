import os
import pickle
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader

OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
LLM_MODEL_NAME = "text-davinci-003"
PDF_FILE_NAME = "story.pdf"
VECTOR_STORE_NAME = "vectorized_doc.pkl"

if os.getenv(OPENAI_API_KEY_NAME, default=None) is None:
    raise Exception(f'create an env variable in your computer with the name {OPENAI_API_KEY_NAME}')

llm = OpenAI(model_name=LLM_MODEL_NAME)

loader = UnstructuredPDFLoader(PDF_FILE_NAME, mode="elements", strategy="fast")
documents = loader.load()

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)

# Save vector store
# TODO - partition the work into different table
with open(VECTOR_STORE_NAME, "wb") as f:
    pickle.dump(vector_store, f)

retriever = vector_store.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)

query = "Where does Sujal current live and where does he work?"
result = qa({"query": query})

print(result['result'])
