import re
import random
import hashlib
import logging
from time import sleep
from datetime import datetime
from itertools import groupby
from operator import itemgetter
from urllib.parse import urljoin
from bs4 import BeautifulSoup as bs
from multiprocessing import cpu_count, Pool

from utils.config import END_YEAR
from newspaper_config.proceso import *
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
    news_text = None

    if "conferencia-mananera-de-amlo" in url:
        news_text = "---video-mananera-amlo---"
    
    if "fotogaleria-" in url:
        news_text = "---fotogaleria---"

    response = get_url(url, method="GET", headers=headers)
    soup = bs(response.content, "lxml")

    if get_date:
        date_div = soup.find("div", {"class":"fecha-y-seccion"})
        full_date_str = date_div.find("div", {"class":"fecha"}).text
        
        # get date from text
        date_str = re.search(r"\w+, (\d{1,2} de \w+ de \d{4}) .*", full_date_str).group(1)
        article_date = datetime.strptime(date_str, "%d de %B de %Y")
    else:
        article_date = None

    if news_text is not None:
        return news_text, article_date

    tags = [tag.text for tag in soup.find_all("a", {"class": "tag label"})]
    if "Cartón" in tags:
        # article is a comic, so there's no text
        return "---carton---", article_date
    
    caption = soup.find("figcaption")
    if caption is not None and ("caricatura" in caption.text or "cartón" in caption.text):
        # article is a comic, so there's no text
        return "---carton---", article_date
    
    # div with the article's data
    main_div = soup.find("article", {"class": "main-article"})
    
    # body
    article = main_div.find("div", {"class": "cuerpo-nota"})
    news_text = "\n".join(
        c.text
        for c in article.children
        if c.name is None or c.name in ["p", "blockquote", "div", "span", "em", "code"]
    )

    # remove unwanted text
    unwanted_regex = [
        r"\[video .*\]\[\/video\]",
        r"\n.*? by .*?on Scribd",
        r"\[\/caption\]\[caption .*?\]",
        r"\[\/caption\]",
        r"\[caption .*?\]",
        r"https:\/\/(twitter|x)\.com\/.*?[ \n]",
        r"Nota relacionada: *",
        r"https:\/\/www\.proceso\.com\.mx\/.*?[ \n]",
        r"\[playlist .*?\]",
        r"https:\/\/www\.youtube\.com\/.*?[ \n]",
        r"https:\/\/www\.facebook\.com\/.*?[ \n]",
        r"https:\/\/www\.(.*?)\.com\/.*?[ \n]",
        r"https:\/\/www\.(.*?)\.com\.mx\/.*?[ \n]",
    ]
    for regex in unwanted_regex:
        news_text = re.sub(regex, "", news_text)
    
    # clean text
    news_text = news_text.replace(u'\xa0', u' ')
    news_text = re.sub("(\n *)+", "\n", news_text)
    news_text = re.sub("\n$", "", news_text)
    news_text = re.sub("^\n", "", news_text)

    if news_text == "" and "video" in url:
        news_text = "---video---"

    return news_text, article_date


def get_article_id(article):

    path = article.find("a")["href"]
    full_url = urljoin(BASE_URL, path)

    return hashlib.md5(full_url.encode("utf8")).hexdigest()


def parse_article(article_str: str, section_name: str, article_id: str) -> tuple[tuple, bool, str]:
    """
    Get all the information about the article

    Parameters
    ----------
    article_str : str
        string of the article's HTML 
    section_name : str
        name of the section that contains the article
    article_id : str
        id of the article

    Returns
    -------
    tuple
        tuple with the file path for the article and the dict with the article data
    bool
        flag that indicates if the article's year is smaller than the END_YEAR
    str
        article id
    """
    article = bs(article_str, "lxml")

    path = article.find("a")["href"]
    full_url = urljoin(BASE_URL, path)

    title = article.find("h2", {"class": "titulo"}).text
    summary = article.find("p", {"class": "resumen"}).text
    
    # get date from url 
    match = re.match(r".*?\/(\d{4}\/\d{1,2}\/\d{1,2})\/.*", path)
    if match is not None:
        url_date_str = match.group(1)
        url_date = datetime.strptime(url_date_str, "%Y/%m/%d")

        # stop processing article
        if url_date.year < END_YEAR:
            return None, True, None
        
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

        # stop processing article
        if get_date and article_date.year < END_YEAR:
            return None, True, None
        
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
        "id": article_id, 
        "newspaper": NEWSPAPER_NAME, 
        "section": section_name.replace("_", " "),
        "date": final_date,
        "url": full_url, 
        "title": title, 
        "summary": summary,
        "text": article_text,
        "error_message": error_message
    }
    
    return (file_path, article_data), False, article_id


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

    articles_params = []
    for article in articles:
        article_id = get_article_id(article)

        if article_id in processed_ids:
            LOGGER.debug(f"Already processed article {article_id}")
        else:
            articles_params.append((str(article), section_name, article_id))

    if len(articles_params) == 0:
        LOGGER.info("All articles have been processed")
        return False, processed_ids

    # process articles in parallel
    with Pool(cpu_count()) as p:
        results = p.starmap(parse_article, articles_params)

    # set flag as True if any of the articles has the exclude flag as True
    end = any(map(itemgetter(1), results))

    # keep only articles where the exclude flag is False
    final_results = [[r[0], r[2]] for r in results if not r[1]]

    if len(final_results) == 0:
        # all the articles have the excluded flag as True
        LOGGER.info(f"No articles with year >= {END_YEAR}")
    else:
        articles_info, article_ids = zip(*final_results)

        # write results
        for file_path, group in groupby(articles_info, itemgetter(0)):
            articles_data = list(map(itemgetter(1), group))
            write_to_json_safe(articles_data, file_path)

        # update processed ids set
        updated_processed_ids = processed_ids.union(set(article_ids))

        # update file with processed ids
        save_processed_ids(NEWSPAPER_NAME, section_name, updated_processed_ids)

    return end, updated_processed_ids


def get_section_data(section_name: str):
    """
    Get all the articles from the given section
    """
    section_id = SECTIONS[section_name]
    subsection_id = SUBSECTIONS[section_name] if section_name in SUBSECTIONS else 0

    processed_ids = get_processed_ids(NEWSPAPER_NAME, section_name)
    LOGGER.debug(f"{len(processed_ids)} processed ids")

    # get starting point
    last_page = get_section_checkpoint(NEWSPAPER_NAME, section_name)
    if last_page is None:
        page_num = 1
    else:
        last_page = int(last_page)
        if last_page < 0:
            LOGGER.info("Section already finished")
            return
        else:
            page_num = last_page + 1

    LOGGER.info(f"Starting from page {page_num}")

    while True:

        if page_num % 100 == 0:
            LOGGER.info(f"page {page_num}")

        # get data
        payload = f"id_seccion={section_id}&id_subseccion={subsection_id}&page={page_num}"
        response = get_url(SEARCH_URL, method="POST", data=payload, headers=HEADERS)
       
        if response.text == "":
            LOGGER.info(f"Finished at page {page_num} because of empty response")
            save_section_checkpoint(NEWSPAPER_NAME, section_name, str(-page_num))
            break
        
        # get all articles
        soup = bs(response.content, "lxml")
        articles = soup.find_all("article")

        final_page, updated_processed_ids = process_page_articles(articles, section_name, processed_ids)

        if final_page:
            LOGGER.info(f"Finished at page {page_num}")
            # save as negative number
            save_section_checkpoint(NEWSPAPER_NAME, section_name, str(-page_num))
            break
        else:
            save_section_checkpoint(NEWSPAPER_NAME, section_name, str(page_num))
        
        # go to next page
        page_num += 1

        # sleep to avoid getting blocked
        if page_num % 20 == 0:
            sleep(random.randint(1,3))

        processed_ids = updated_processed_ids


def scrape_proceso():
    for section_name in SECTIONS:
        LOGGER.info(f"Getting {section_name} data")

        try:
            get_section_data(section_name)
        except Exception:
            LOGGER.error(f"Error getting data from {section_name} section", exc_info=True)


if __name__ == "__main__":
    scrape_proceso()