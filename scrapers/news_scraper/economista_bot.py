import json
import random
import hashlib
import logging
import pandas as pd
from time import sleep
from datetime import datetime
from unidecode import unidecode
from bs4 import BeautifulSoup as bs
from multiprocessing.pool import ThreadPool

from utils.config import END_YEAR
from newspaper_config.economista import *
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


def get_text(url: str) -> tuple[str, str]:
    """
    Get the article's full text

    Parameters
    ----------
    url : str
        url to the article

    Returns
    -------
    str
        text of the article
    """

    response = get_url(url, method="GET", headers=HEADERS)
    soup = bs(response.content, "lxml")
    
    news_content = soup.find("script", {"type": "application/ld+json"})
    if news_content is not None:
        # get json with data
        article_dict = json.loads(news_content.text)
        
        summary_text = article_dict["description"]
        news_text = article_dict["articleBody"]
    else:
        # summary
        summary = soup.find("div", {"class": "resumeNew"})
        if summary is None:
            body_div = soup.find("div", {"class": "newsbody"})
            summary = body_div.find_previous_sibling("div")

        summary_text = summary.text

        # body
        article = soup.find("div", {"id": "readNote"})
        if article is None:
            article_div = soup.find("div", {"class": "newsbody"})
            article = article_div.find_all("div", recursive=False)[1].find_all("div", recursive=False)[0]

        news_text = article.text
    
    return news_text, summary_text


def get_text_parallel(url: str) -> tuple[str, str, str]:
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
        article_text, summary_text = get_text(url)
    except Exception as ex:
        error_message = str(ex)
        article_text = summary_text = None
    else:
        error_message = None

    return article_text, summary_text, error_message


def hash_url(url: str):
    return hashlib.md5(url.encode("utf8")).hexdigest()


def parse_articles(articles: list) -> pd.DataFrame:
    """
    Get all the information about the article

    Parameters
    ----------
    articles : list
        list with the articles data

    Returns
    -------
    pd.DataFrame
        data frame with the formatted articles
    """
    
    articles_df = pd.json_normalize(articles)

    # rename columns
    articles_df = articles_df.rename(columns=RENAME_COLUMNS)

    # cast columns
    articles_df["date"] = articles_df.date.astype(float)/1000
    articles_df["date"] = articles_df.date.apply(datetime.fromtimestamp)

    # set constant columns
    articles_df["newspaper"] = NEWSPAPER_NAME.replace("_", " ")

    # build columns
    articles_df["id"] = articles_df.url.apply(hash_url)
    articles_df["file_path"] = articles_df.date.apply(lambda d: d.strftime("%Y/%m.json"))
    articles_df["date"] = articles_df.date.apply(lambda d: d.strftime("%Y-%m-%dT%H:%M:%S"))
    articles_df["section"] = articles_df.section.apply(unidecode).apply(str.lower)

    articles_df = articles_df[[
        "id",
        "newspaper",
        "section",
        "date",
        "url",
        "title",
        "file_path"
    ]]
    
    return articles_df


def process_batch_articles(articles: list, processed_ids: set, section_name: str) -> tuple[bool, set]:
    """
    Process all the articles in a page and save their data and ids to the respective files

    Parameters
    ----------
    articles : list
        list with the soup objects of the articles
    processed_ids : set
        set with the ids of the articles that have already been processed
    section_name : str
        name of the section that is being processed

    Returns
    -------
    bool
        flag to indicate if this should be the final batch or not
    set
        updated processed ids set with the ids from this batch
    """
    end = False
    LOGGER.debug(f"{len(articles)} raw articles")

    articles_df = parse_articles(articles)
    LOGGER.debug(f"{articles_df.shape[0]} unfiltered articles")

    # get the oldest record in the df
    min_date = datetime.strptime(articles_df.date.min(), "%Y-%m-%dT%H:%M:%S")
    LOGGER.debug(f"min_date: {min_date}")
    if min_date.year < END_YEAR:
        end = True

        # filter df to keep only the needed articles
        articles_df = articles_df.loc[articles_df.date >= f"{END_YEAR}-01-01"]

    LOGGER.debug(f"{articles_df.shape[0]} after date filter")

    # remove articles that were already saved
    articles_df = articles_df.loc[~articles_df.id.isin(processed_ids)]
    LOGGER.debug(f"{articles_df.shape[0]} after ids filter")

    if articles_df.shape[0] == 0:
        LOGGER.info("All articles have been processed")
        return end, processed_ids

    # get articles text concurrently
    with ThreadPool(NUM_THREADS) as p:
        text_results = p.map(get_text_parallel, articles_df.url.tolist())

    articles_text, articles_summary, error_messages = zip(*text_results)

    # save results into df
    articles_df["summary"] = list(articles_summary)
    articles_df["text"] = list(articles_text)
    articles_df["error_message"] = list(error_messages)

    # save results
    updated_processed_ids = processed_ids
    for file_path, group in articles_df.groupby("file_path"):
        group = group.drop(columns=["file_path"])
        
        # write articles
        articles_data = group.to_dict(orient="records")
        write_to_json_safe(articles_data, file_path)

        # update processed ids set
        updated_processed_ids = updated_processed_ids.union(set(group.id))

        # update file with processed ids
        save_processed_ids(NEWSPAPER_NAME, section_name, updated_processed_ids)

    return end, updated_processed_ids


def get_section_data(section_name: str):
    """
    Get all the articles from the given section
    """

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
        if page_num % 50 == 0:
            LOGGER.info(f"batch {page_num}")

        # get data
        params = {
            "section": section_name,
            "order": "user-modification-date desc",
            "size": BATCH_SIZE,
            "page": page_num
        }
        response = get_url(SEARCH_URL, method="GET", params=params, headers=HEADERS)
       
        # raise exception if status != 200
        response.raise_for_status()
        
        response = response.json()
        final_batch, updated_processed_ids = process_batch_articles(response["items"], processed_ids, section_name)

        if final_batch:
            LOGGER.info(f"Finished at page {page_num}")
            break
        else:
            save_section_checkpoint(NEWSPAPER_NAME, section_name, str(page_num))

        if "next" not in response:
            LOGGER.info("Ingested no next page")
            break

        # go to next page
        page_num += 1

        # sleep to avoid getting blocked
        if page_num % 20 == 0:
            sleep(random.randint(1,3))
        
        processed_ids = updated_processed_ids

    # save negative offset to indicate that this section is done
    save_section_checkpoint(NEWSPAPER_NAME, section_name, str(-page_num))


def scrape_el_economista():
    for section_name in SECTIONS:
        LOGGER.info(f"Getting {section_name} data")

        try:
            get_section_data(section_name)
        except Exception:
            LOGGER.error(f"Error getting data from {section_name} section", exc_info=True)


if __name__ == "__main__":
    scrape_el_economista()