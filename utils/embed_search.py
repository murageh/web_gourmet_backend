import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

with open('../examples/openai.html', 'r') as file:
    web_content = file.read()


def parse_html(html_string):
    soup = BeautifulSoup(html_string, 'html.parser')
    return soup


web_content = parse_html(web_content).text

# an example question about the 2022 Olympics
query = 'What does this website talk about?'

prompt = query = f"""Below is an excerpt from a website. Answer the question based on the information provided. If you 
cannot find the answer, answer with "I cannot find the answer."

Article:
\"\"\"
{web_content}
\"\"\"

Question: {query}
"""

response = client.chat.completions.create(
    messages=[
        {'role': 'system',
         'content': 'You answer questions about nay given website, provided with information from it.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODEL,
    temperature=0,
)

print(response.choices[0].message.content)
