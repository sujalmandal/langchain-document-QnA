import os
import uuid
import pickle
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader

OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
LLM_MODEL_NAME = "text-davinci-003"

if os.getenv(OPENAI_API_KEY_NAME, default=None) is None:
    raise Exception(f'create an env variable in your computer with the name {OPENAI_API_KEY_NAME}')

llm = OpenAI(model_name=LLM_MODEL_NAME)


def read_document(file_name):
    # read file

    document_id = uuid.uuid4()
    loader = UnstructuredPDFLoader(file_name, mode="elements", strategy="fast")
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    vectorized_file_name = f'vec_{document_id}.pkl'
    # Save vector store
    with open(vectorized_file_name, "wb") as f:
        pickle.dump(vector_store, f)

    return document_id


def answer(document_id, question):
    vector_store = pickle.load(open(f'vec_{document_id}.pkl', "rb"))
    retriever = vector_store.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True)

    result = qa({"query": question})

    return result['result']


def show_all_files():
    file_names = []
    for path, currentDirectory, files in os.walk("./"):
        for file in files:
            if file.startswith("vec_"):
                file_names.append(file)

    return file_names
