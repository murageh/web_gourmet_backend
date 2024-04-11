import os

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from client import client


# Parse the HTML string using Beautiful Soup
def parse_html(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    return soup


# Extract relevant information from the HTML using Beautiful Soup
def extract_information(html_soup):
    extracted_information = []
    for element in html_soup.find_all():
        if element.name == 'title':
            extracted_information.append(element.text.strip())
        elif element.name == 'p':
            extracted_information.append(element.text.strip())
        # Add more conditions to extract other relevant information
    return extracted_information


def text_to_embeddings(text_corpus):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(text_corpus)
    print("Embeddings shape:", embeddings.shape)  # Debug print
    return embeddings, vectorizer


# Calculate cosine similarity between query and corpus embeddings
def calculate_similarity(query_embeddings, corpus_embeddings):
    similarity_scores = cosine_similarity(query_embeddings, corpus_embeddings)
    return similarity_scores


# Retrieve similar documents based on similarity scores
def retrieve_similar_documents(query, corpus, vectorizer, top_n=5):
    query_embedding = vectorizer.transform([query])
    corpus_embeddings = vectorizer.transform(corpus)

    similarity_scores = calculate_similarity(query_embedding, corpus_embeddings)
    ranked_indices = similarity_scores.argsort(axis=1)[:, ::-1]

    top_documents = [corpus[i] for i in ranked_indices.ravel()[:top_n]]

    return top_documents


# Answer questions using OpenAI's GPT model
def answer_question_with_gpt(context, question):
    prefix = "Given the following information from a website:\n"
    q = prefix + context + "\nQuestion: " + question + "\nAnswer:"
    resp = client.embeddings.create(model="text-embedding-3-small",
                                    input=q,
                                    )
    print(resp.data)
    return resp.data[0].embedding


if __name__ == "__main__":
    with open('./examples/openai.html', 'r') as file:
        html_string = file.read()

    # Parse HTML string
    html_soup = parse_html(html_string)

    # Extract information from HTML
    corpus = extract_information(html_soup)

    # Get context from website content
    context = ". ".join(corpus)

    # Get prompt from user
    # prompt = input("Enter your prompt: ")
    question = 'what is openai?'

    # Convert text to embeddings and get vectorizer
    embeddings, vectorizer = text_to_embeddings(corpus)

    # Retrieve similar documents based on prompt
    similar_documents = retrieve_similar_documents(question, corpus, vectorizer)
    print("Similar documents:")
    for idx, doc in enumerate(similar_documents, 1):
        print(f"{idx}. {doc}")
    print('-' * 50)

    # while True:
    #     # Get question from user
    #     question = input("Enter your question (or type 'quit' to exit): ")
    #
    #     if question.lower() == 'quit':
    #         break
    #
    #     # Generate response using GPT
    #     response = answer_question_with_gpt(context, question)
    #     print("GPT Response:", response)

    # Generate response using GPT
    response = answer_question_with_gpt(context, question)
    print("GPT Response:", response)
