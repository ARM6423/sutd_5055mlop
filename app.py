# QUESTION: Implement a web UI frontend for your chatbot that you can demo in class.

#--- ADD YOUR SOLUTION HERE (40 points)---
# Finetuned model + RAG chatbot



import streamlit as st
import random
import time
import requests
import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document, BaseRetriever
import fitz
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import re
from typing import Any, List
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask_cors import CORS
from flask import Flask, request, jsonify

st.title("SUTD Chatbot")

if "messages" not in st.session_state:
  st.session_state.messages = []

if "vector_store" not in st.session_state:
  if "downloaded_docs" not in os.listdir():
    os.system("unzip downloaded_docs.zip")
  download_folder = "downloaded_docs"

  # Clean documents to remove whitespace + newline
  def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

  # Process PDF
  def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

  documents = []
  pdf_files = [
    os.path.join(download_folder, f)
    for f in os.listdir(download_folder)
    if f.lower().endswith(".pdf")
  ]
  for file in pdf_files:
    text = extract_text_from_pdf(file)
    text = clean_text(text)
    document = Document(page_content=text)
    documents.append(document)
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50
  )

  split_docs = text_splitter.split_documents(documents)
  embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
  st.session_state.vector_store = FAISS.from_documents(split_docs, embedding_model)
  st.session_state.vector_store.save_local("faiss_index")

if "model" not in st.session_state:
  hf_token = ""
  login(token=hf_token)

  finetuned_model_name = "{YOUR_HF_NAME}/llama-3.2-1B-sutdqa"
  finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)
  finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
  finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_name)
  st.session_state.model = pipeline("text-generation", model=finetuned_model, tokenizer=finetuned_tokenizer, device_map="auto", max_new_tokens=512, return_full_text=False)


for message in st.session_state.messages:
  with st.chat_message(message["role"]):
      st.markdown(message["content"])

if query := st.chat_input("Ask a question about SUTD"):
  st.session_state.messages.append({"role": "user", "content": query})
  with st.chat_message("user"):
      st.markdown(query)

  with st.chat_message("assistant"):
    dense_results = st.session_state.vector_store.similarity_search(query, k=20)
    candidate_texts = [doc.page_content for doc in dense_results]
    tokenized_candidates = [word_tokenize(text.lower()) for text in candidate_texts]
    tokenized_query = word_tokenize(query.lower())
    bm25 = BM25Okapi(tokenized_candidates)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

    context_chunks = []
    for rank, idx in enumerate(top_indices):
        context_chunks.append(candidate_texts[idx])
    context = "\n".join(context_chunks)

    prompt = (
    # "Use the following pieces of context to answer the question at the end. "
    # "If you don't know the answer, just say that you don't know, don't try to make up an answer. If the context is somewhat relevant, make your best effort to give a coherent answer.\n\n"
    """You are an expert educational advisor for the Singapore University of Technology and Design (SUTD).
      Answer the following question as if you are helping a prospective student who is curious about SUTD.
      Be clear, accurate, friendly, and informative.
      Provide a helpful, clear, and concise answer for the following question"""
    f"Context:\n{context}\n\n"
    f"Question: {query}\n"
    "Answer:"
    )
    response = st.session_state.model(prompt, max_new_tokens=512)[0]["generated_text"]
    st.write(response)

  st.session_state.messages.append({"role": "assistant", "content": response})