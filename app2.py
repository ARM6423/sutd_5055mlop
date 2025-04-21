# Finetuned model chatbot -> for viewing code only

#%%writefile app2.py

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
    prompt = (
    # "Use the following pieces of context to answer the question at the end. "
    # "If you don't know the answer, just say that you don't know, don't try to make up an answer. If the context is somewhat relevant, make your best effort to give a coherent answer.\n\n"
    """You are an expert educational advisor for the Singapore University of Technology and Design (SUTD).
      Answer the following question as if you are helping a prospective student who is curious about SUTD.
      Be clear, accurate, friendly, and informative.
      Provide a helpful, clear, and concise answer for the following question"""
    f"Question: {query}\n"
    "Answer:"
    )
    response = st.session_state.model(prompt, max_new_tokens=512)[0]["generated_text"]
    st.write(response)

  st.session_state.messages.append({"role": "assistant", "content": response})