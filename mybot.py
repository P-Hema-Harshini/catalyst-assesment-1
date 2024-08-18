# RUN THIS CODE IN GOOGLE COLAB

!pip install -q --upgrade google-generativeai langchain-google-genai python-dotenv
!pip install langchain
!pip install pafy
!pip install youtube-transcript-api
!pip install youtube_dl
!pip install -U langchain-community
import os
import warnings
from pathlib import Path as p
from pprint import pprint

import pandas as pd
import pafy
from youtube_transcript_api import YouTubeTranscriptApi

from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS, Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_youtube_transcript(youtube_url):
    video_id = youtube_url.split("v=")[1]
    transcript_list = YouTubeTranscriptApi.get_transcripts(video_id)
    transcript = ""
    for transcript_part in transcript_list:
        transcript += transcript_part.fetch()["text"]
    return transcript

def load_and_split_youtube_transcript(youtube_url):
    transcript = get_youtube_transcript(youtube_url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=0)
    texts = text_splitter.split_text(transcript)
    return texts

def convert_texts_to_vectors(texts, google_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_index = Chroma.from_texts(texts, embeddings)
    return vector_index

prompt_template = """
  Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
  provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
  Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def answer_question(youtube_url, question, google_api_key):
    texts = load_and_split_youtube_transcript(youtube_url)
    vector_index = convert_texts_to_vectors(texts, google_api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    docs = vector_index.get_relevant_documents(question)
    response = chain({"input_documents":docs, "question": question}, return_only_outputs=True)
    answer = response
    return answer

import urllib.parse
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url):
    """Extract the video ID from a YouTube URL"""
    parsed_url = urllib.parse.urlparse(url)
    video_id = urllib.parse.parse_qs(parsed_url.query).get('v')[0]
    return video_id

def get_transcript(video_id):
    """Get the transcript of a YouTube video"""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([line["text"] for line in transcript])
    return transcript_text

def answer_question(transcript, question):
    """Answer a question based on the transcript"""
    # Implement your question answering logic here
    # For now, just return a dummy answer
    return "Dummy answer"

youtube_url = "https://www.youtube.com/watch?v=Pj0neYUp9Tc"
video_id = extract_video_id(youtube_url)

question = "What is the video about?"
transcript = get_transcript(video_id)
answer = answer_question(transcript, question)
print(answer)

!pip install gradio

import nltk

def answer_question(transcript, question):
    """Answer a question based on the transcript"""
    # Extract named entities from the transcript
    tokens = nltk.word_tokenize(transcript)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.ne_chunk(tagged)

    # Find entities that match the question
    question_words = nltk.word_tokenize(question)
    question_tags = nltk.pos_tag(question_words)
    matches = []
    for entity in entities:
        if type(entity) == nltk.tree.Tree:
            for child in entity:
                if child[1] == "NNP" and child[0].lower() in [word.lower() for word in question_words]:
                    matches.append(entity)

    # Return the matched entities as the answer
    if matches:
        answer = " ".join([child[0] for child in matches])
    else:
        answer = "I'm not sure."

    return answer

import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id):
    """Get the transcript of a YouTube video"""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([line["text"] for line in transcript])
    return transcript_text

def answer_question(transcript, question):
    """Answer a question based on the transcript"""
    # Implement your question answering logic here
    # For now, just return a dummy answer
    return "Dummy answer"

def interface(video_url, question):
    video_id = video_url.split("=")[1]
    transcript = get_transcript(video_id)
    answer = answer_question(transcript, question)
    return answer

iface = gr.Interface(
    fn=interface,
    inputs=["text", "text"],
    outputs="text",
    title="YouTube Video Q&A",
    description="Enter a YouTube video URL and a question, and I'll try to answer it based on the video transcript."
)

iface.launch()