import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama

load_dotenv()
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"))
pdf_path = "https://arxiv.org/pdf/2210.03629" # ReAct: Synergizing Reasoning and Acting in Language Models
pdf_path = "https://arxiv.org/pdf/2405.04517" # xLSTM: Extended Long Short-Term Memory
pdf_path = "https://arxiv.org/pdf/2305.14314" # QLoRA: Efficient Fine-tuning of Quantized LLMs

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def summarize_paper(docs):
    prompt = PromptTemplate(
        template="Summarize the following paper: {docs}", input_variables=["docs"]
    )
    chain = prompt | llm
    return chain.invoke(docs)

def main():
    docs = load_pdf(pdf_path)
    # print(docs)
    response = summarize_paper(docs)
    print(response.content)

if __name__ == "__main__":
    main()
