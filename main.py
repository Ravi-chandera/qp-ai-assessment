from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA




DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """You are a highly experienced individual in the filed of disaster handling and management.
You have a bachelors degree in disaster management and you have a masters and phd in the same filed. You can predict the timeline
of the events. Also you are accuracte all the time about your prediction and for that you got many awards for your work towards disaster prediction.
You have published various research papers and people recognitize you for your contribution towards disaster management.

Context: {context}
Question: {question}

try keeping information most useful technically.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5,
        device="cuda"  # Use "cuda" to enable CUDA
    )
    return llm

#QA Model Function
def qa_bot():

  embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2",
      model_kwargs={'device': 'cuda'}
  )

  db = FAISS.load_local(DB_FAISS_PATH, embeddings)
  llm = load_llm()
  qa_prompt = set_custom_prompt()

  qa = retrieval_qa_chain(llm, qa_prompt, db)
  return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response
