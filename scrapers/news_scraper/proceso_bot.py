import re
import locale
import random
import hashlib
import logging
import requests
from time import sleep
from datetime import datetime
from urllib.parse import urljoin
from bs4 import BeautifulSoup as bs

from utils.config import END_YEAR
from utils.methods import get_processed_ids, save_processed_ids, write_to_json
from utils.proceso_config import BASE_URL, HEADERS, NEWSPAPER_NAME, SEARCH_URL, SECTIONS, SUBSECTIONS

# set locale language to ES
locale.setlocale(locale.LC_TIME, "es_ES")

# setup loggers
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
critical_logs = ["urllib3", "charset_normalizer"]
for logger_name in critical_logs:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def get_text(url: str, get_date: bool=False) -> tuple[str, datetime]:
    """
    Get the article's full text

    Parameters
    ----------
    url : str
        url to the article
    get_date : bool, optional
        indicates if we should also get the date, by default False

    Returns
    -------
    tuple[str, datetime]
        text of the article
        date of the article if get_date is True, else None
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
    }
    response = requests.get(url, headers=headers)
    soup = bs(response.content, "lxml")
    
    # div with the article's data
    main_div = soup.find("article", {"class": "main-article"})
    
    # body
    article = main_div.find("div", {"class": "cuerpo-nota"})
    news_text = "\n".join(c.text for c in article.children if c.name == "p")
    
    # clean text
    news_text = news_text.replace(u'\xa0', u' ')
    news_text = re.sub("(\n *)+", "\n", news_text)
    news_text = re.sub("\n$", "", news_text)
    
    if get_date:
        date_div = soup.find("div", {"class":"fecha-y-seccion"})
        full_date_str = date_div.find("div", {"class":"fecha"}).text
        
        # get date from text
        date_str = re.search(r"\w+, (\d{1,2} de \w+ de \d{4}) .*", full_date_str).group(1)
        article_date = datetime.strptime(date_str, "%d de %B de %Y")
    else:
        article_date = None

    return news_text, article_date


def get_article_id(article):

    path = article.find("a")["href"]
    full_url = urljoin(BASE_URL, path)

    return hashlib.md5(full_url.encode("utf8")).hexdigest()


def parse_article(article, section_name: str) -> dict:
    """
    Get all the information about the article
    """
    path = article.find("a")["href"]
    full_url = urljoin(BASE_URL, path)

    title = article.find("h2", {"class": "titulo"}).text
    summary = article.find("p", {"class": "resumen"}).text
    
    # get date from url 
    match = re.match(r".*?\/(\d{4}\/\d{1,2}\/\d{1,2})\/.*", path)
    if match is not None:
        url_date_str = match.group(1)
        url_date = datetime.strptime(url_date_str, "%Y/%m/%d")
        
        get_date = False
    else:
        # try to get date from the article
        get_date = True
    
    try:
        article_text, article_date = get_text(full_url, get_date)
    except Exception as ex:
        LOGGER.error(f"Error getting {full_url} text", exc_info=True)
        error_message = str(ex)
        article_text = None
    else:
        error_message = None
        
    if url_date is not None:
        final_date = url_date.strftime("%Y-%m-%d")
        file_path = url_date.strftime("%Y/%m.json")

    elif article_date is not None:
        final_date = article_date.strftime("%Y-%m-%d") 
        file_path = article_date.strftime("%Y/%m.json")

    else:
        final_date = None
        file_path = url_date.strftime("na_date.json")
    
    # build final dictionary
    article_data = {
        "newspaper": NEWSPAPER_NAME, 
        "section": section_name,
        "date": final_date,
        "url": full_url, 
        "title": title, 
        "summary": summary,
        "text": article_text,
        "error_message": error_message
    }
    
    return article_data, file_path


def process_page_articles(articles: list, section_name: str, processed_ids: set) -> bool:
    """
    Process all the articles in a page and save their data and ids to the respective files

    Parameters
    ----------
    articles : list
        list with the soup objects of the articles
    section_name : str
        name of the section we're processing
    processed_ids : set
        set with the ids of the articles that have already been processed

    Returns
    -------
    bool
        flag to indicate if this should be the final page or not
    """
    end = False

    page_data = {}
    for article in articles:
        article_id = get_article_id(article)

        if article_id in processed_ids:
            LOGGER.debug(f"Already processed article {article_id}")
            continue
        else:
            processed_ids.add(article_id)

        article_data, file_path = parse_article(article, section_name)

        # check if the article is still in a year we want
        article_year = int(file_path.split("/")[0])
        if article_year < END_YEAR:
            end = True
            continue

        # save id
        article_data["id"] = article_id

        # save into final dict
        if file_path in page_data:
            page_data[file_path].append(article_data)
        else:
            page_data[file_path] = [article_data]
    
    # write results
    for file_path, articles_data in page_data.items():
        write_to_json(articles_data, file_path)

    # update file with processed ids
    save_processed_ids(NEWSPAPER_NAME, section_name, processed_ids)

    return end

def get_section_data(section_name: str):
    """
    Get all the articles from the given section
    """
    section_id = SECTIONS[section_name]
    subsection_id = SUBSECTIONS[section_name] if section_name in SUBSECTIONS else 0

    processed_ids = get_processed_ids(NEWSPAPER_NAME, section_name)
    LOGGER.debug(f"{len(processed_ids)} processed ids")

    if section_name == "nacional" and len(processed_ids) == 0:
        raise Exception("Nacional IDS not found!!!")
    
    page_num = 1
    while True:

        if page_num % 100 == 0:
            LOGGER.info(f"page {page_num}")

        # get data
        payload = f"id_seccion={section_id}&id_subseccion={subsection_id}&page={page_num}"
        response = requests.post(SEARCH_URL, data=payload, headers=HEADERS)
       
        if response.status_code != 200:
            LOGGER.info(f"Finished at page {page_num}")
            break
        
        # get all articles
        soup = bs(response.content, "lxml")
        articles = soup.find_all("article")

        final_page = process_page_articles(articles, section_name, processed_ids)

        if final_page:
            LOGGER.info(f"Finished at page {page_num}")
            break
        
        # go to next page
        page_num += 1

        # sleep to avoid getting blocked
        sleep(random.randint(1,3))


def scrape_proceso():
    for section_name in SECTIONS:
        LOGGER.info(f"Getting {section_name} data")

        try:
            get_section_data(section_name)
        except Exception:
            LOGGER.error(f"Error getting data from {section_name} section", exc_info=True)


if __name__ == "__main__":
    scrape_proceso()