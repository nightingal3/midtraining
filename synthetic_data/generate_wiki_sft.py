import requests
import csv
import openai
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from collections import deque
import multiprocessing
from tqdm import tqdm
import random
from prompts import SYSTEM_PROMPTS, FIRST_USER_PROMPTS

# You'll need to set up your OpenAI API key
openai.api_key = 'your-api-key-here'

def get_category_members(category):
    """
    Fetch all members (pages and subcategories) of a given category.
    """
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "max",
        "cmtype": "page|subcat"
    }

    members = []
    while True:
        response = requests.get(base_url, params=params)
        data = response.json()

        if 'query' in data:
            members.extend(data['query']['categorymembers'])

        if 'continue' not in data:
            break

        params['cmcontinue'] = data['continue']['cmcontinue']

    return members

def breadth_first_search(categories, max_depth=4, min_depth=2):
    """
    Perform a breadth-first search through category hierarchy to find articles.
    """
    articles = []
    visited_categories = set()
    category_queue = deque([(cat, 0) for cat in categories])  # (category, depth)

    with tqdm(desc="Searching categories", unit=" categories") as pbar:
        while category_queue:
            category, depth = category_queue.popleft()

            if category in visited_categories or depth > max_depth:
                continue

            visited_categories.add(category)
            members = get_category_members(category)

            for member in members:
                if member['ns'] == 0 and depth >= min_depth:  # This is an article
                    articles.append((f"https://en.wikipedia.org/?curid={member['pageid']}", depth))
                elif member['ns'] == 14 and depth < max_depth:  # This is a subcategory
                    category_queue.append((member['title'][9:], depth + 1))  # Remove "Category:" prefix

            pbar.update(1)

    return articles

def scrape_wikipedia_article(url):
    """Scrape the main text content of a Wikipedia article."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find(id="mw-content-text").find_all('p')
    return ' '.join([p.text for p in content])

def chunk_text(text, chunk_size=1000):
    """Break the text into chunks of approximately chunk_size characters."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    for sentence in sentences:
        if current_size + len(sentence) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(sentence)
        current_size += len(sentence)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def generate_questions(chunk):
    """Generate quiz questions based on the given text chunk using an LLM."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates quiz questions based on given text."},
            {"role": "user", "content": f"Generate a multiple-choice question based on this text:\n\n{chunk}"}
        ]
    )
    return response.choices[0].message['content']

def process_article(article_url):
    """Process a single article to generate questions."""
    content = scrape_wikipedia_article(article_url)
    chunks = chunk_text(content)
    questions = []
    for chunk in chunks:
        question = generate_questions(chunk)
        questions.append((article_url, question))
        if len(questions) >= 1:  # Limit to 1 question per article
            break
    return questions

def main():
    nltk.download('punkt')  # Download the punkt tokenizer for sentence splitting

    #base_categories = ["Physics", "Biology", "Chemistry", "Mathematics", "World_history", "Literature"]
    base_categories = ["Physics"]
    articles = breadth_first_search(base_categories, max_depth=7, min_depth=2)
    breakpoint()
    print(f"Total articles found: {len(articles)}")
    
    # Shuffle the articles to get a random selection
    random.shuffle(articles)
    
    # Select the first max_articles
    max_articles = 1000  # Adjust this number as needed
    selected_articles = articles[:max_articles]
    
    # Use multiprocessing to generate questions
    with multiprocessing.Pool() as pool:
        all_questions = list(tqdm(
            pool.imap(process_article, [article[0] for article in selected_articles]),
            total=len(selected_articles),
            desc="Generating questions"
        ))

    # Flatten the list of questions
    questions = [q for sublist in all_questions for q in sublist]

    # Write questions to CSV
    with open('wikipedia_quiz_questions.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Article URL', 'Question'])
        for question in questions:
            writer.writerow(question)

    print(f"Generated {len(questions)} questions from {len(selected_articles)} articles.")

if __name__ == "__main__":
    main()