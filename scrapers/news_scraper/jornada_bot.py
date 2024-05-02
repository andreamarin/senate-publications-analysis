import re
import json
import hashlib
import logging
import pandas as pd
from unidecode import unidecode
from urllib.parse import urljoin
from bs4 import BeautifulSoup as bs
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool

from utils.config import END_YEAR
from newspaper_config.jornada import *
from utils.methods import get_processed_ids, get_section_checkpoint, get_url, save_processed_ids, save_section_checkpoint, write_to_json_safe

# setup loggers
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
critical_logs = ["urllib3", "charset_normalizer", "filelock"]
for logger_name in critical_logs:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def get_article_data(url: str):
    response = get_url(url, method="GET")
    soup = bs(response.content, "lxml")

    news_content = soup.find("script", {"type": "application/ld+json"})
    if news_content is not None:
        clean_content = re.sub(r"([\t\n] *)+", "", news_content.text)

        # get json with data
        article_dict = json.loads(clean_content)

        summary_text = article_dict["description"]
        title = re.sub(r"^ *La Jornada:", "", article_dict["headline"])
    else:
        title = soup.find("div", {"class": "cabeza"}).text
        summary_text = None
 
    news_text = soup.find("div", {"id": "article-text"}).text

    # replace unwanted data
    for class_name in ["pie-foto", "credito-autor", "credito-titulo", "hemero"]:
        replace_div = soup.find("div", {"class": class_name})

        if replace_div is not None:
            replace_text = replace_div.text.strip()
            if len(replace_text) > 0:
                replace_text = f"\n *{replace_text} *\n"
                news_text = re.sub(replace_text, "\n", news_text)
    
    # replace multiple new lines with just one
    news_text = re.sub("(\n *)+", "\n", news_text)

    # remove leading \n and spaces
    news_text = news_text.strip("\n").strip()

    article_data = {
        "url": url, 
        "title": title.strip(),
        "summary": summary_text.strip(), 
        "text": news_text.strip()
    }
    
    return article_data

def get_articles_parallel(url: str) -> tuple[dict, str]:
    """
    Wrapper to call the `get_text` function in parallel and catch any errors

    Parameters
    ----------
    url : str
        article's url

    Returns
    -------
    str
        text of the article, if there was an error it's None
    str
        article's summary, if there was an error it's None
    str
        error message if there was an error getting the text, else None
    """
    try:
        article_data = get_article_data(url)
    except Exception as ex:
        article_data = {
            "url": url, 
            "title": None,
            "summary": None,
            "text": None,
             "error_message": str(ex)
        }
    else:
        article_data["error_message"] = None

    return article_data


def hash_url(url: str):
    return hashlib.md5(url.encode("utf8")).hexdigest()


def parse_section_articles(section_articles: list, date_url: str):
    """
    Get all the data from the articles in a section
    """
    
    # list with articles urls
    article_urls = [
        urljoin(date_url, article.find("a")["href"])
        for article in section_articles
        if "feet" not in article["class"] and article.find("a", {"class": "cabeza"}) is not None 
    ]

    LOGGER.debug(f"Processing {len(article_urls)} articles")

    # get articles text concurrently
    with ThreadPool(NUM_THREADS) as p:
        article_results = p.map(get_articles_parallel, article_urls)

    return list(article_results)


def get_section_data(section_url: str, date: datetime, section_name: str, processed_ids: set):

    response = get_url(section_url, method="GET")
    soup = bs(response.content, "lxml")

    articles_div = soup.find("div", {"id": "section-cont"})
    if articles_div is None:
        # section is a single article
        LOGGER.debug("Section has a single article")
        article_data = get_article_data(section_url)
        articles_results = [article_data] 
    else:
        # get all the articles in the section
        section_articles = articles_div.find_all("div", recursive=False)
        date_url = urljoin(BASE_URL, date.strftime("%Y/%m/%d/"))
        articles_results = parse_section_articles(section_articles, date_url)

    articles_df = pd.DataFrame(articles_results)

    # add missing columns
    articles_df["id"] = articles_df.url.apply(hash_url)
    articles_df["newspaper"] = NEWSPAPER_NAME.replace("_", " ")
    articles_df["section"] = section_name
    articles_df["date"] = date.strftime("%Y-%m-%d")

    # remove articles that were already saved
    articles_df = articles_df.loc[~articles_df.id.isin(processed_ids)]

    if articles_df.shape[0] == 0:
        LOGGER.info("Section already processed")
        return processed_ids
        
    # write articles to json
    file_path = date.strftime("%Y/%m.json")
    articles_data = articles_df.to_dict(orient="records")
    write_to_json_safe(articles_data, file_path)

    # update processed ids set
    updated_processed_ids = processed_ids.union(set(articles_df.id))

    # update file with processed ids
    ids_file_path = date.strftime("%Y/%m/%d")
    save_processed_ids(NEWSPAPER_NAME, ids_file_path, updated_processed_ids)

    return updated_processed_ids


def get_date_articles(date: datetime):
    date_data_file = date.strftime("%Y/%m/%d")

    processed_ids = get_processed_ids(NEWSPAPER_NAME, date_data_file)
    LOGGER.debug(f"Already processed {len(processed_ids)} articles")

    processed_sections_str = get_section_checkpoint(NEWSPAPER_NAME, date_data_file)
    if processed_sections_str is None:
        processed_sections_str = "---"

    if FINISHED_STR in processed_sections_str:
        LOGGER.info("Date already finished")
        return

    date_url = urljoin(BASE_URL, date.strftime("%Y/%m/%d/"))
    LOGGER.debug(date_url)

    response = get_url(date_url, method="GET")

    if response.status_code == 404 or response.status_code == 403:
        LOGGER.info("No data for current date")
        save_section_checkpoint(NEWSPAPER_NAME, date_data_file, FINISHED_STR)
        return
    
    soup = bs(response.content, "lxml")

    sections = soup.find("div", {"class": "main-sections"}).find_all("td")
    for section in sections:
        if "class" in section.attrs and section["class"] == "sflinktd":
            continue

        section_name = unidecode(section.text).lower()

        if section_name in EXCLUDE_SECTIONS or section_name == "":
            LOGGER.debug(f"Skipping section {section_name}")
            continue

        if f"---{section_name}---" in processed_sections_str:
            LOGGER.info(f"Section {section_name} already processed")
            continue

        LOGGER.info(f"Parsing section {section_name}")
        section_url = urljoin(BASE_URL, section.find("a")["href"])

        try:
            updated_processed_ids = get_section_data(section_url, date, section_name, processed_ids)
        except Exception:
            LOGGER.warning(f"Couldn't parse section {section_name}", exc_info=True)
        else:
            processed_ids = updated_processed_ids

            # update string with processed sections
            processed_sections_str += f"{section_name}---"
            save_section_checkpoint(NEWSPAPER_NAME, date_data_file, processed_sections_str)

    # add string that indicates that the date is finished
    processed_sections_str += FINISHED_STR
    save_section_checkpoint(NEWSPAPER_NAME, date_data_file, processed_sections_str)


def scrape_la_jornada():
    end_date = datetime.now()

    last_date = get_section_checkpoint(NEWSPAPER_NAME, DATE_CHECKPOINT_FILE)
    if last_date is None:
        start_date = datetime(END_YEAR, 1, 1)
    else:
        start_date = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)

    LOGGER.info(f"Starting from date {start_date} until {end_date}")

    date = start_date
    while date <= end_date:
        LOGGER.info(f"Date: {date}")

        get_date_articles(date)

        date_str = date.strftime("%Y-%m-%d")
        save_section_checkpoint(NEWSPAPER_NAME, DATE_CHECKPOINT_FILE, date_str)

        date = date + timedelta(days=1)

        LOGGER.info("====="*10)


if __name__ == "__main__":
    scrape_la_jornada()