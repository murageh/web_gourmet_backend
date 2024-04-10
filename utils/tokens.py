import ast
import re
from typing import List

import pandas as pd
import requests
import tiktoken
from bs4 import BeautifulSoup
from openai import OpenAI
from scipy import spatial

from client import client

GPT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

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
]


def sections_from_page(
        page: BeautifulSoup,
        sections_to_ignore=None,
) -> list[tuple[list[str], str]]:
    """
    From any page, return a flattened list of all nested subsections.
    *WE ARE USING BS4*
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children)
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

    return sections_inner


def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    """
    Clean a section's text.
    Removes any text in square brackets, which are often citations.
    Remove newlines and extra spaces.
    """
    parent_titles, text = section
    cleaned_text = text.strip()
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
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string


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


def fetch_html_from_url(url: str) -> str:
    """Fetch HTML content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return ""


def fetch_html_from_file(file_path: str) -> str:
    """Fetch HTML content from a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""


def embed_text(strings: list[str], client: OpenAI = None) -> list[list[float]]:
    embeddings = []
    for batch_start in range(0, len(strings), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = strings[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end-1}")
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response.data):
            assert i == be.index  # double check embeddings are in the same order as input
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings


def tokenize(sections: list[tuple[list[str], str]]) -> List[str]:
    # split sections into chunks
    MAX_TOKENS = 1600
    page_strings = []
    for section in sections:
        section = clean_section(section)
        section_strings = split_strings_from_subsection(section, max_tokens=MAX_TOKENS)
        page_strings.extend(section_strings)

    return page_strings


def generate_embeddings(site: str, strings=None):
    if strings is None:
        strings = []
    from client import client
    embeddings = embed_text(strings, client=client)
    df = pd.DataFrame({"text": strings, "embedding": embeddings})
    print(df)

    # save to CSV
    save_path = site + "_embeddings.csv"
    df.to_csv(save_path, index=False)


# ACTUAL TEXT SEARCH
def prepare_data(site: str):
    # site is the name of the website, without the .com or .co.ke, and without the www. or https://
    df = pd.read_csv('./' + site + "_embeddings.csv")
    return df


# search function
def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100,
        client: OpenAI = client,
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = ('Use the below information extracted from a website to answer the subsequent question. If the '
                    'answer cannot be found in the sections, write "I could not find an answer."')
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
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about any website, provided information about it."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message



if __name__ == "__main__":
    # url = "https://murageh.co.ke"
    # html = fetch_html_from_url(url)
    # page = BeautifulSoup(html, 'html.parser')
    # sections = sections_from_page(page)
    # strings = tokenize(sections)
    # generate_embeddings("murageh.co.ke", strings)

    # prepare_data("murageh.co.ke")
    df = prepare_data("murageh.co.ke")
    # convert embeddings from CSV str type back to list type
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    # print(df)

    query = "is there an email available to reach the website's owner?"
    res = ask(
        query=query,
        df=df,
        model=GPT_MODEL,
        token_budget=4096 - 500,
        print_message=True,
    )
    print(res)
