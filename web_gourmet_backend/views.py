import ast
import os

import pandas as pd
from bs4 import BeautifulSoup
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from utils.embed_search_v2_latest import validate_website, strip_website, fetch_html_from_url, \
    generate_embeddings_and_save, \
    ask, sections_from_page, tokenize, is_data_stale, strip_date_from_filename

websites = {}  # dictionary to store the websites. once a user submits a website, the website is stored here.


def store_website(site, embeddings):
    """
    This function stores the website in the website dictionary
    :param site: Website name. It has the date appended to it after the '__'.
                This is used to determine if the embeddings need to be updated
    :param embeddings:
    :return:
    """
    try:
        stale = is_data_stale(site)
        print(f"is {site} stale? {stale}")
    except Exception as e:
        print("Failed to check if the data is stale", e)
        return

    site = strip_date_from_filename(site)

    websites[site] = {
        'embeddings': embeddings,
        'stale': stale,
    }


# on startup, load previous embeddings
def load_previous_embeddings(specific: str = None):
    """
    This function loads the previous embeddings from the file tree.
    Load the embeddings directory and load the embeddings for each website.
    Each embedding is stored as a csv file as <website_id>_embeddings.csv
    :return:
    """
    path = 'embeddings'

    for file in os.listdir(path):
        if specific and specific != strip_date_from_filename(file):
            continue
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            df['embedding'] = df['embedding'].apply(ast.literal_eval)
            store_website(file, df)


def print_loaded_embeddings():
    print('websites:', len(websites))
    for key, value in websites.items():
        print(key, '->', 'value')


# load the previous embeddings
load_previous_embeddings()
print('loaded websites:', len(websites))
# print_loaded_embeddings()


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
def submit_url(request):
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
    original_url = request.POST.get('url')
    url = original_url
    if not url:
        return JsonResponse(UserResponse(400, "Please specify a website.").to_json(), status=400)

    # validate the website
    valid = validate_website(url)
    if not valid:
        return JsonResponse(UserResponse(400, "Invalid website.").to_json(), status=400)

    # check if the website has been submitted before, then check if the data is stale
    try:
        url = strip_website(url)
        if url in websites:
            if not websites[url]['stale']:
                return JsonResponse(UserResponse(200, "Website submitted successfully.").to_json())
    except Exception as e:
        print("Failed to check if the website is stale", e)
        return JsonResponse(UserResponse(500, "Failed to submit the website.").to_json(), status=500)

    # invoke the scraper
    html_content = fetch_html_from_url(original_url)
    if not html_content:
        return JsonResponse(UserResponse(
            500,
            "Failed to fetch the website content. Make sure you input the entire "
            "website. Including \"https://\"").to_json(),
                            status=500,
                            )

    # generate embeddings
    try:
        page = BeautifulSoup(html_content, 'html.parser')
        sections = sections_from_page(page)
        strings = tokenize(sections)
        _, save_path = generate_embeddings_and_save(url, strings)
    except Exception as e:
        print(e)
        return JsonResponse(UserResponse(500, "Failed to generate embeddings for the website.").to_json(), status=500)

    # update website store
    load_previous_embeddings(specific=url)
    print_loaded_embeddings()

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
    original_url = request.POST.get('url')
    url = strip_website(original_url)
    if not url:
        return JsonResponse(UserResponse(400, "Please specify a website.").to_json(), status=400)

    # fetch the question from the request
    question = request.POST.get('question')
    if not question:
        return JsonResponse(UserResponse(400, "Please specify a question.").to_json(), status=400)

    # fetch embeddings for the site
    try:
        df = websites[url]['embeddings']
    except KeyError:
        return JsonResponse(UserResponse(400, f"({url}) not found. Did you already submit it?").to_json(), status=400)
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
