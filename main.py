import os
import uuid
import pickle

from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader

OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
LLM_MODEL_NAME = "text-davinci-003"
memory_cache = {}

llm = OpenAI(model_name=LLM_MODEL_NAME, temperature=0.4)


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

    if document_id not in memory_cache:
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        memory_cache[document_id] = memory

    general_system_template = """
    You are 'dda-docgpt', an AI assistant that reads documents and answers questions.
    
    Use the following pieces of context and the document you read to answer the user's question. 
    ----------------
    {context}
    ----------------
    
    Follow the rules below while generating responses.
    ----------------
    1. If the user is not asking a question, reply with "Please keep the conversation about the document.".
    2. If you do not understand the question or cannot find the answer, say 'I am unable to answer your question.'.
    ----------------
    
    Answer in the following format.
    ------------
    dda-docgpt: <your_answer>
    ------------
    
    Here are some examples
    ------------
    user: thank you
    dda-docgpt: you're welcome
    ------------
    user: thanks
    dda-docgpt: glad to help
    ------------
    user: ok
    dda-docgpt: Please keep the conversation about the document.
    ------------
    user: hi
    dda-docgpt: Hello. Ask your questions.
    ------------
    user: how are you
    dda-docgpt: I am good. Ask your questions.
    ------------
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # Create the multipurpose chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory_cache[document_id],
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )

    result = qa({"question": question})

    return result['answer']


def show_all_files():
    file_names = []
    for path, currentDirectory, files in os.walk("./"):
        for file in files:
            if file.startswith("vec_"):
                file_names.append(file)

    return file_names
