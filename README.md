# Web Gourmet - Backend

## Description
Web Gourmet is a web application that allows users to input a website URL, then proceed to ask any questions they may 
have about the website. 
The application will then provide a response to the user's question. The application is built
using the Django framework and the Python programming language. 
The application uses the BeautifulSoup library to scrape the website and extract the information needed to answer 
the user's question. The application leverages the power of AI for the question-answering functionality. 
This is achieved using the embedded-search capabilities of openAI's GPT models.

## Installation
To install the application, you will need to have Python installed on your machine. You can download Python from the
official website. Once you have Python installed, you can clone the repository and install the required dependencies
using the following commands:
```bash
git clone
cd web-gourmet
pip install -r requirements.txt
```
After installing the dependencies, you can run the application using the following command:
```bash
python manage.py runserver
```
The application will be accessible at http:// localhost:8000.

Proceed to then start the frontend, which will be accessible in the root project. More instructions on how to run it 
are available on the project's `README.md` file.

## Usage
To use the application, you can input a website URL in the input field on the homepage. You can then proceed to ask any
questions you may have about the website. The application will provide a response to your question. You can ask
questions such as "What is the website about?" or "What services does the website offer?" The application will use the
information extracted from the website to answer your question.

## Contributing
If you would like to contribute to the project, you can fork the repository and submit a pull request with your changes.
You can also open an issue if you have any questions or suggestions for the project.

## License
This project is licensed under the MIT License [LICENSE](LICENSE).