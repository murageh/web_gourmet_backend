import ast
import json
import os

import pandas as pd
from bs4 import BeautifulSoup
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import csrf_protect
from utils.embed_search_v2 import validate_website, strip_website, fetch_html_from_url, generate_embeddings_and_save, \
    ask, sections_from_page, tokenize

websites = {}  # dictionary to store the websites. once a user submits a website, the website is stored here.


def store_website(site, embeddings):
    """
    This function stores the website in the website dictionary
    :param site:
    :param embeddings:
    :return:
    """
    websites[site] = {
        'embeddings': embeddings
    }


# on startup, load previous embeddings
def load_previous_embeddings():
    """
    This function loads the previous embeddings from the file tree.
    Load the embeddings directory and load the embeddings for each website.
    Each embedding is stored as a csv file as <website_id>_embeddings.csv
    :return:
    """
    path = 'embeddings'
    for file in os.listdir(path):
        if file.endswith(".csv"):
            website_id = file.split('_')[0]
            df = pd.read_csv(os.path.join(path, file))
            df['embedding'] = df['embedding'].apply(ast.literal_eval)
            store_website(website_id, df)


def print_loaded_embeddings():
    print('websites:', len(websites))
    for key, value in websites.items():
        print(key, '->', value)


# load the previous embeddings
load_previous_embeddings()


class UserResponse:
    def __init__(self, status, message):
        self.status = status
        self.message = message
        self.error = None

    def to_json(self):
        # should return json
        resp = {
            'status': self.status,
            'message': self.message,
            'error': self.error
        }
        return resp


def index(request):
    return JsonResponse(UserResponse(200, "Welcome to Web Gourmet API.").to_json())


@csrf_exempt
def submit_website(request):
    """
    This function is called when a user submits a website to be scraped.
    This will invoke the scraper, and generate the embeddings for the website content
    before returning a success message to the user.
    If any error occurs, it will return an error message to the user.
    (This is the first step,and it forms a basis for consecutive calls to the API)
    :param request:
    :return:
    """
    if request.method != 'POST':
        return JsonResponse(UserResponse(404, f"Cannot ${request.method} to this path").to_json(), status=404)

    # fetch the website from the request
    site = request.POST.get('site')
    if not site:
        return JsonResponse(UserResponse(400, "Please specify a website.").to_json(), status=400)

    # validate the website
    valid = validate_website(site)
    if not valid:
        return JsonResponse(UserResponse(400, "Invalid website.").to_json(), status=400)

    # invoke the scraper
    html_content = fetch_html_from_url(site)
    if not html_content:
        return JsonResponse(UserResponse(500, "Failed to fetch the website content.").to_json(), status=500)

    # generate embeddings
    try:
        page = BeautifulSoup(html_content, 'html.parser')
        sections = sections_from_page(page)
        strings = tokenize(sections)
        embeddings = generate_embeddings_and_save(site, strings)
    except Exception as e:
        print(e)
        return JsonResponse(UserResponse(500, "Failed to generate embeddings for the website.").to_json(), status=500)

    # store website
    store_website(strip_website(site), embeddings)

    # return success message
    return JsonResponse(UserResponse(200, "Website submitted successfully.").to_json())


@csrf_exempt
def ask_question(request):
    """
    This function is called when a user asks a question.
    We first check to see if the user has submitted the website before.
    If the user has not submitted the website, we return an error message.
    If the user has submitted the website, we generate the embeddings for the question
    and compare it with the embeddings of the website content.
    We then answer the question and return the answer to the user.
    :param request:
    :return:
    """
    if request.method != 'POST':
        return JsonResponse(UserResponse(404, f"Cannot ${request.method} to this path").to_json(), status=404)

    # fetch the website from the request
    site = request.POST.get('site')
    site = strip_website(site)
    if not site:
        return JsonResponse(UserResponse(400, "Please specify a website.").to_json(), status=400)

    # fetch the question from the request
    question = request.POST.get('question')
    if not question:
        return JsonResponse(UserResponse(400, "Please specify a question.").to_json(), status=400)

    # fetch embeddings for the site
    try:
        df = websites[site]['embeddings']
    except KeyError:
        return JsonResponse(UserResponse(400, "Website not found.").to_json(), status=400)
    except Exception as e:
        print(e)
        return JsonResponse(UserResponse(500, "Failed to fetch the website embeddings.").to_json(), status=500)

    # ask the question
    answer = ask(
        query=question,
        df=df,
        print_message=True,
    )
    if not answer:
        return JsonResponse(UserResponse(500, "Failed to answer the question.").to_json(), status=500)

    # return the answer
    return JsonResponse(UserResponse(200, answer).to_json())
