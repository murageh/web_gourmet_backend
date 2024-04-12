import os
import re
from typing import List

import pandas as pd
import requests
import tiktoken
from bs4 import BeautifulSoup
from openai import OpenAI
from scipy import spatial

from utils.app_logger import logger
from utils.client import client as openai_client
from utils.configs import GPT_MODEL, EMBEDDING_MODEL, BATCH_SIZE, MAX_TOKENS

SECTIONS_TO_IGNORE = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Footnotes",
    "Bibliography",
    "Sources",
    "Citations",
    "Literature",
    "Footnotes",
    "Notes and references",
    "Photo gallery",
    "Works cited",
    "Photos",
    "Gallery",
    "Notes",
    "References and sources",
    "References and notes"
]  # These are common sections that are often not useful for answering questions. This list can be expanded/modified.


def sections_from_page(
        page: BeautifulSoup,
        sections_to_ignore=None,
) -> list[tuple[list[str], str]]:
    """
     Extracts nested subsections from a web page using BeautifulSoup.

     Args:
         page: BeautifulSoup object representing the parsed HTML content.
         sections_to_ignore (list, optional): List of section titles to exclude. Defaults to None.

     Returns:
         List[Tuple[List[str], str]]: A list of tuples where each tuple represents a subsection.
             The first element is a list of parent section titles, and the second element is the text content.
     """
    if sections_to_ignore is None:
        sections_to_ignore = SECTIONS_TO_IGNORE
    sections_inner = []

    headings = page.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

    for heading in headings:
        if heading.name == "h1":
            parent_titles = [heading.text.strip()]
        else:
            parent_titles = [heading.text.strip()]
            parent = heading
            while parent := parent.find_previous_sibling():
                if parent.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    parent_titles.insert(0, parent.text.strip())
        if parent_titles[-1] in sections_to_ignore:
            continue
        text = ""
        sibling = heading.find_next_sibling()
        while sibling and sibling.name not in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            text += sibling.text.strip() + " "
            sibling = sibling.find_next_sibling()
        sections_inner.append((parent_titles, text.strip()))

    # special case where there are no headings
    # -> uses the entire page as a single section (paragraphs, spans, etc.)
    if not sections_inner:
        text = ""
        for child in page.children:
            text += child.text.strip() + " "
        sections_inner.append(([page.title.text.strip()], text.strip()))

    return sections_inner


def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    """"
    Cleans a section's text by removing bracketed citations and extra whitespace.

    Args:
        section (Tuple[List[str], str]): Tuple representing a subsection.
            The first element is a list of parent section titles, and the second element is the text content.

    Returns:
        Tuple[List[str], str]: The cleaned subsection with citations removed and whitespace trimmed.
    """
    parent_titles, text = section
    text.strip()
    cleaned_text = re.sub(r"\[\d+]", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n*", " ", cleaned_text)
    return parent_titles, cleaned_text


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def truncated_string(
        s: str,
        model: str,
        max_tokens: int,
        print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(s)
    truncated_str = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_str


def halved_by_delimiter(s: str, delimiter: str = "\n") -> list[str]:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = s.split(delimiter)
    if len(chunks) == 1:
        return [s, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(s)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]


def split_strings_from_subsection(
        subsection: tuple[list[str], str],
        max_tokens: int = 1000,
        model: str = GPT_MODEL,
        max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    titles, text = subsection
    s = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(s)

    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [s]

    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(s, model=model, max_tokens=max_tokens)]

    # otherwise, split in half and recurse
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results

    # otherwise, no split was found, so just truncate (should be very rare)
    return [truncated_string(s, model=model, max_tokens=max_tokens)]


def validate_website(site: str) -> bool:
    """
   Validates a website URL.

   Args:
       site (str): The website URL to validate.

   Returns:
       bool: True if the URL is valid, False otherwise.
   """
    if not site:
        return False
    url_pattern = r"^https?://[\w\-\.]+\.[a-z]{2,}(?:/[^\s]*)*$"
    return bool(re.match(url_pattern, site))


def strip_website(site: str):
    """
    This function strips the website to remove any unnecessary characters.
    :param site:
    :return:
    """
    if not site:
        raise ValueError("Please provide a website to scrape.")
    site = site.strip()
    if site.startswith("http://"):
        site = site.replace("http://", "")
    if site.startswith("https://"):
        site = site.replace("https://", "")
    if site.endswith("/"):
        site = site[:-1]
    site = site.replace('/', '_')
    return site


def fetch_html_from_url(url: str) -> str:
    """
    Fetches HTML content from a URL with error handling.

    Args:
        url (str): The URL to fetch.

    Returns:
        str: The fetched HTML content or an empty string if an error occurs.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for non-2xx status codes
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL: {e}")
        return ""


def fetch_html_from_file(file_path: str) -> str:
    """Fetch HTML content from a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""


def embed_text(strings: list[str], client: OpenAI = None, batch_size=BATCH_SIZE) -> list[list[float]]:
    """
    Embeds a list of text strings using OpenAI's API with batching and logging.

    Args:
        strings (List[str]): The list of text strings to embed.
        client (OpenAI, optional): The OpenAI client object. Defaults to None.
        batch_size (int, optional): The batch size for sending requests to the API. Defaults to 1000.

    Returns:
        List[List[float]]: A list of lists containing the embeddings for each text string.
    """
    embeddings = []
    for batch_start in range(0, len(strings), batch_size):
        batch_end = batch_start + batch_size
        batch = strings[batch_start:batch_end]
        logger.info(f"Batch {batch_start} to {batch_end - 1}")
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response.data):
            assert i == be.index  # double check embeddings are in the same order as input
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings


def tokenize(sections: list[tuple[list[str], str]]) -> List[str]:
    # split sections into chunks
    page_strings = []
    for section in sections:
        section = clean_section(section)
        section_strings = split_strings_from_subsection(section, max_tokens=MAX_TOKENS)
        page_strings.extend(section_strings)

    return page_strings


def get_current_date_str():
    """
    Return the current date in ISO format. (YYYY-MM-DDTHH:MM:SS...)
    """
    from datetime import datetime
    return datetime.now().isoformat()


def append_date_to_filename(filename: str):
    """
    Appends the current date in ISO format to a filename.

    Args:
        filename (str): The original filename.

    Returns:
        str: The filename with the date appended, separated by '__'.
    """
    datestr = get_current_date_str()
    base, ext = os.path.splitext(filename)
    return f"{base}__{datestr}{ext}"


def strip_date_from_filename(filename: str):
    """
    Removes the date portion from a filename, assuming it's in ISO format and separated by '__'.

    Args:
        filename (str): The filename with potentially a date appended.

    Returns:
        str: The filename without the date.
    """
    parts = filename.split("__")
    if len(parts) > 1:
        return "__".join(parts[:-1])
    else:
        return filename


def get_date_str_from_filename(filename: str):
    """Get the date from a filename."""
    base, _ = os.path.splitext(filename)
    datestr = base.split("__")[-1]
    return datestr


def get_date_from_filename(filename: str):
    """Get the date from a filename."""
    datestr = get_date_str_from_filename(filename)
    from datetime import datetime
    return datetime.fromisoformat(datestr)


def is_data_stale(filename: str, days: int = 7):
    """
    Check if the data in filename is stale, based on the date in the filename.
    Returns True if the data is older than (or equal to) days.
    :param filename: The filename to check. Should have a date in ISO format at the end, separated by '__'.
    :param days: The number of days after which the data is considered stale. Default is 7 days.
    """
    date1 = get_date_from_filename(filename)
    from datetime import datetime
    date2 = datetime.now()
    delta = date2 - date1
    return delta.days >= days


def generate_embeddings_and_save(site: str, strings=None):
    if strings is None:
        strings = []
    embeddings = embed_text(strings, client=openai_client)
    df = pd.DataFrame({"text": strings, "embedding": embeddings})

    # create embeddings directory if it doesn't exist
    os.makedirs(f"embeddings/", exist_ok=True)

    # save to CSV
    save_path = "embeddings/" + strip_website(site) + ".csv"

    # append date to filename
    save_path = append_date_to_filename(save_path)

    df.to_csv(save_path, index=False)
    print(f"Embeddings saved to {save_path}")

    return df, save_path


# ACTUAL TEXT SEARCH
def prepare_data(site: str):
    """
    This function reads the embeddings from the file system and returns a dataframe
    :param site: name of the website, without the .com or .co.ke, and without the www. Or https://
    :return:
    """
    df = pd.read_csv('../embeddings/' + site + "_embeddings.csv")
    return df


# search function
def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100,
        client: OpenAI = openai_client,
) -> tuple[list[str], list[float]]:
    """
    Ranks strings by their relatedness to a query using cosine similarity.

    Args:
        query (str): The query string.
        df (pd.DataFrame): Dataframe containing text and embedding columns.
        relatedness_fn (function, optional): Function to calculate relatedness. Defaults to cosine similarity.
        top_n (int, optional): The number of top-ranked strings to return. Defaults to 100.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing the ranked strings and their relatedness scores.
    """
    query_embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"])) for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    top_n_strings_and_relatednesses = strings_and_relatednesses[:top_n]
    return zip(*top_n_strings_and_relatednesses)


def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """
    Creates a message for GPT with relevant source texts, considering token budget.

    Args:
        query (str): The query string.
        df (pd.DataFrame): Dataframe containing text and embedding columns.
        model (str): The GPT model identifier.
        token_budget (int): The maximum number of tokens allowed for the message.

    Returns:
        str: The formatted message for GPT.
    """
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = (
        'Use the below information extracted from a website to answer the subsequent question. If the '
        'answer cannot be found in the sections, write "I could not find an answer."'
    )
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWebsite section:\n"""\n{string}\n"""'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
        query: str,
        df: pd.DataFrame = None,
        model: str = GPT_MODEL,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        pass
    messages = [
        {"role": "system", "content": "You answer questions about any website, provided information about it."},
        {"role": "user", "content": message},
    ]
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message


if __name__ == "__main__":
    """
    Basically, this is how you would use the functions in this file.
    """
    # url = "https://crispice.murageh.co.ke"
    # html = fetch_html_from_url(url)
    # page = BeautifulSoup(html, 'html.parser')
    # sections = sections_from_page(page)
    # strings = tokenize(sections)
    # generate_embeddings_and_save("murageh.co.ke", strings)

    # -----------------------

    # prepare_data("murageh.co.ke")
    # df = prepare_data("murageh.co.ke")
    # # convert embeddings from CSV str type back to list type
    # df['embedding'] = df['embedding'].apply(ast.literal_eval)
    # # print(df)
    #
    # query = "is there an email available to reach the website's owner?"
    # res = ask(
    #     query=query,
    #     df=df,
    #     model=GPT_MODEL,
    #     token_budget=4096 - 500,
    #     print_message=True,
    # )
    # print(res)

    pass
