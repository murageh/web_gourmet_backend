import ast
import os

import pandas as pd
from bs4 import BeautifulSoup
from django.core.exceptions import ValidationError
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt

from utils.embed_search_v2_latest import (
    validate_website, strip_website, fetch_html_from_url,
    generate_embeddings_and_save,
    ask, sections_from_page, tokenize, is_data_stale, strip_date_from_filename
)
from utils.app_logger import logger

websites = {}  # dictionary to store website data (embeddings, staleness)


def store_website(website, embeddings):
    """
    Stores the website data (embeddings and staleness flag) in the websites dictionary.
    """
    try:
        is_stale = is_data_stale(website)
        website = strip_date_from_filename(website)
        websites[website] = {
            'embeddings': embeddings,
            'is_stale': is_stale,
        }
    except Exception as e:
        # Log the error
        logger.error(f"Failed to store website data: {e}")


def load_previous_embeddings(specific_website: str = None):
    """
    Loads previously generated website embeddings from the 'embeddings' directory.
    """
    path = 'embeddings'
    for filename in os.listdir(path):
        if filename.endswith(".csv") and (
                not specific_website or specific_website == strip_date_from_filename(filename)):
            try:
                df = pd.read_csv(os.path.join(path, filename))
                df['embedding'] = df['embedding'].apply(ast.literal_eval)
                store_website(filename, df)
            except Exception as e:
                # Log the error
                logger.error(f"Failed to load embeddings for {filename}: {e}")


def print_loaded_embeddings():
    print('websites:', len(websites))
    for key, value in websites.items():
        print(key, '->', 'value')


load_previous_embeddings()


class UserResponse:
    def __init__(self, status: int, message: str = None):
        self.status = status
        self.message = message
        self.error = None

    def to_json(self):
        return {
            'status': self.status,
            'message': self.message,
            'error': self.error
        }


def index(request):
    return JsonResponse(UserResponse(200, "Welcome to Web Gourmet API.").to_json())


@csrf_exempt
def submit_url(request: HttpRequest) -> JsonResponse:
    """
    Handles user requests to submit a website URL for scraping and embedding generation.

    Returns a success message or error response depending on the outcome.
    """
    if request.method != 'POST':
        return JsonResponse(UserResponse(404, f"Method '{request.method}' not allowed").to_json(), status=404)

    original_url = request.POST.get('url')

    if not original_url:
        return JsonResponse(UserResponse(400, "Please specify a website.").to_json(), status=400)

    try:
        if not validate_website(original_url):
            raise ValidationError("Invalid website URL.")

        website_url = strip_website(original_url)

        if website_url in websites and not websites[website_url]['is_stale']:
            return JsonResponse(UserResponse(200, "Website already submitted. Ask me a question.").to_json())

    except ValidationError as e:
        return JsonResponse(UserResponse(400, "Invalid website URL.").to_json(), status=400)
    except Exception as e:
        return JsonResponse(UserResponse(500, f"Failed to validate website URL: {e}").to_json(), status=500)

    # Fetch and process website content
    html_content = fetch_html_from_url(original_url)
    if not html_content:
        return JsonResponse(UserResponse(
            500,
            "Failed to fetch the website content. Make sure you input the entire "
            "website. Including \"https://\"").to_json(),
                            status=500,
                            )

    try:
        page = BeautifulSoup(html_content, 'html.parser')
        sections = sections_from_page(page)
        strings = tokenize(sections)
        _, save_path = generate_embeddings_and_save(website_url, strings)
    except Exception as e:
        logger.error(f"Failed to generate embeddings for {website_url}: {e}")
        return JsonResponse(UserResponse(500, "Failed to generate embeddings.").to_json(), status=500)

    # Update website store and reload embeddings
    load_previous_embeddings(specific_website=website_url)

    return JsonResponse(UserResponse(200, "Website submitted successfully.").to_json())


@csrf_exempt
def ask_question(request: HttpRequest) -> JsonResponse:
    """
    This function is called when a user asks a question.
    We first check to see if the user has submitted the website before.
    If the user has not submitted the website, we return an error message.
    If the user has submitted the website, we generate the embeddings for the question
    and compare it with the embeddings of the website content.
    We then use the retrieved similar content to answer the question and return the answer to the user.
    :param request:
    :return:
    """
    if request.method != 'POST':
        return JsonResponse(UserResponse(404, f"Method '{request.method}' not allowed").to_json(), status=404)
    website_url = strip_website(request.POST.get('url'))
    question = request.POST.get('question')

    if not website_url:
        return JsonResponse(UserResponse(400, "Please specify a website URL.").to_json(), status=400)

    if not question:
        return JsonResponse(UserResponse(400, "Please specify a question.").to_json(), status=400)

    try:
        website_data = websites[website_url]
    except KeyError:
        return JsonResponse(UserResponse(400, f"Website '{website_url}' not found.").to_json(), status=400)
    except Exception as e:
        logger.error(f"Failed to fetch website data for {website_url}: {e}")
        return JsonResponse(UserResponse(500, "Internal server error.").to_json(), status=500)

    answer = ask(query=question, df=website_data['embeddings'], print_message=True)
    if not answer:
        return JsonResponse(UserResponse(500, "Failed to answer the question.").to_json(), status=500)

    return JsonResponse(UserResponse(200, answer).to_json())
