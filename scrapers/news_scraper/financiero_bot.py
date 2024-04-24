import json
import random
import logging
import numpy as np
import pandas as pd
from time import sleep
from datetime import datetime
from unidecode import unidecode
from bs4 import BeautifulSoup as bs
from multiprocessing.pool import ThreadPool

from utils.config import END_YEAR
from newspaper_config.financiero import *
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


def get_text(url: str) -> str:
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

    response = get_url(url, method="GET")
    soup = bs(response.content, "lxml")
    
    # body
    article = soup.find("article", {"class": "article-body-wrapper"})
    news_text = "\n".join(c.text for c in article.children if c.name != "article")
    
    return news_text


def get_text_parallel(url: str) -> tuple[str, str]:
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
        error message if there was an error getting the next, else None
    """
    try:
        article_text = get_text(url)
    except Exception as ex:
        error_message = str(ex)
        article_text = None
    else:
        error_message = None

    return article_text, error_message


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
    articles_df["date"] = pd.to_datetime(articles_df.date)

    # set constant columns
    articles_df["newspaper"] = NEWSPAPER_NAME.replace("_", " ")

    # build columns
    articles_df["url"] = BASE_URL + articles_df["websites.elfinanciero.website_url"]
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
        "summary",
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

    articles_df = parse_articles(articles)

    # get the oldest record in the df
    min_date = datetime.strptime(articles_df.date.min(), "%Y-%m-%dT%H:%M:%S")
    if min_date.year < END_YEAR:
        end = True

        # filter df to keep only the needed articles
        articles_df = articles_df.loc[articles_df.date >= f"{END_YEAR}-01-01"]

    # remove articles that were already saved
    articles_df = articles_df.loc[~articles_df.id.isin(processed_ids)]

    if articles_df.shape[0] == 0:
        LOGGER.info("All articles have been processed")
        return end, processed_ids

    # get articles text concurrently
    with ThreadPool(NUM_THREADS) as p:
        text_results = p.map(get_text_parallel, articles_df.url.tolist())

    articles_text, error_messages = zip(*text_results)

    # save results into df
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
    last_offset = get_section_checkpoint(NEWSPAPER_NAME, section_name)
    if last_offset is None:
        offset = 0
    else:
        last_offset = int(last_offset)
        if last_offset < 0:
            LOGGER.info("Section already finished")
            return
        else:
            offset = last_offset + BATCH_SIZE

    LOGGER.info(f"Starting from offset {offset}")

    total_results = np.inf
    while offset < total_results:
        batch_num = offset / BATCH_SIZE + 1
        if batch_num % 100 == 0:
            LOGGER.info(f"batch {batch_num}")

        # get data
        section_url = section_name.replace("-", "_")
        query = {
            "excludeSections":"",
            "feature":"results-list",
            "feedOffset": offset,
            "feedSize": BATCH_SIZE,
            "includeSections":f"/{section_url}"
        }
        params = {
            "query": json.dumps(query),
            **BASE_PARAMS
        }
        response = get_url(SEARCH_URL, method="GET", params=params)
       
        # raise exception if status != 200
        response.raise_for_status()
        
        response = response.json()

        if np.isinf(total_results):
            # update total results with real number
            total_results = response["count"]
            LOGGER.info(f"{total_results} total results for section")

        final_batch, updated_processed_ids = process_batch_articles(response["content_elements"], processed_ids, section_name)

        if final_batch:
            LOGGER.info(f"Finished at batch {batch_num}")
            break
        else:
            save_section_checkpoint(NEWSPAPER_NAME, section_name, str(offset))
        
        # go to next batch
        offset += BATCH_SIZE

        # sleep to avoid getting blocked
        if batch_num % 20 == 0:
            sleep(random.randint(1,3))

        processed_ids = updated_processed_ids

    # save negative offset to indicate that this section is done
    save_section_checkpoint(NEWSPAPER_NAME, section_name, str(-offset))


def scrape_el_financiero():
    for section_name in SECTIONS:
        LOGGER.info(f"Getting {section_name} data")

        try:
            get_section_data(section_name)
        except Exception:
            LOGGER.error(f"Error getting data from {section_name} section", exc_info=True)


if __name__ == "__main__":
    scrape_el_financiero()