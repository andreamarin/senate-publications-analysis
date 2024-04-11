import re
import json
import locale
import random
import logging
import requests
import numpy as np
import pandas as pd
from time import sleep
from datetime import datetime
from operator import itemgetter
from urllib.parse import urljoin
from bs4 import BeautifulSoup as bs

from utils.config import END_YEAR
from utils.methods import get_processed_ids, get_section_checkpoint, get_url, save_processed_ids, save_section_checkpoint, write_to_json_safe
from utils.animal_politico_config import *

# set locale language to ES
locale.setlocale(locale.LC_TIME, "es_ES")

# setup loggers
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
critical_logs = ["urllib3", "charset_normalizer", "filelock"]
for logger_name in critical_logs:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def get_text(content: str) -> str:
    """
    Get the article's full text
    """
    soup = bs(content, "lxml")
    return soup.text


def get_url_text(url: str) -> str:
    """
    Get article text using the url
    """

    try:
        response = get_url(url)
    except Exception:
        LOGGER.warning(f"Couldn't get url {url}", exc_info=True)
        return None

    # build bs object
    soup = bs(response.content, "lxml")

    text = ""
    for post_details in soup.find_all("div", {"class":"post-details"}):
        
        attatchment = post_details.find("figure")
        if attatchment is not None:
            # section is an attatchment (image, video, tweet)
            continue
        
        details_text = post_details.text
        if re.match(r"^(Lee|Entérate)( (más|también))? *[:\|].*", details_text) is not None:
            # text is a redirect to other article
            continue
            
        text += details_text
        text += "\n"

    return text


def build_url(row: pd.Series) -> str:
    """
    Build the article's url
    """
    if row.categoryPrimarySlug != "":
        article_slug = f"{row.section_slug}/{row.categoryPrimarySlug}/{row.slug}"

    elif row.section == "hablemos_de":
        # get all the categories
        section_categories = set(map(itemgetter("slug"), map(itemgetter("node"), row.categorasDeHablemosDe["edges"])))
        # see which of them is a valid subcategory for the url
        category_slug = SUBCATEGORIES[row.section].intersection(section_categories).pop()
        
        article_slug = f"{row.section_slug}/{category_slug}/{row.slug}"

    elif row.section == "analisis":

        if row.blogSlug == "blog-invitado":
            article_slug = f"{row.section_slug}/invitades/{row.slug}"
        elif row.blogAuthor is None:
            article_slug = f"{row.section_slug}/autores/{row.blogSlug}/{row.slug}"
        else:
            article_slug = f"{row.section_slug}/organizaciones/{row.blogSlug}/{row.slug}"

    else:
        article_slug = f"{row.section_slug}/{row.slug}"
        
    url = urljoin(BASE_URL, article_slug)
    
    return url


def parse_articles(articles: list, section_name: str) -> pd.DataFrame:
    """
    Get all the information about the article

    Parameters
    ----------
    articles_df : list
        string of the article's HTML 
    section_name : str
        name of the section that contains the article

    Returns
    -------
    pd.DataFrame
        df with final columns
    """
    articles_df = pd.DataFrame(articles)

    # set constant columns
    articles_df["newspaper"] = NEWSPAPER_NAME.replace("_", " ")
    articles_df["section"] = section_name
    articles_df["section_slug"] = section_name.replace("_", "-")

    # build columns
    articles_df["url"] = articles_df.apply(build_url, axis=1)
    articles_df["file_path"] = articles_df.date.apply(
        lambda d: datetime.strptime(d, "%Y-%m-%dT%H:%M:%S").strftime("%Y/%m.json")
    )

    # get text from contentRendered where available
    content_condition = (~pd.isna(articles_df.contentRendered)) & (articles_df.contentRendered != "")
    articles_df.loc[content_condition, "text"] = articles_df.loc[content_condition].contentRendered.apply(get_text)

    # get text scraping the url for those who dont have content
    url_condition = pd.isna(articles_df.text)
    articles_df.loc[url_condition, "text"] = articles_df.loc[url_condition].url.apply(get_url_text)

    # set propper section name
    articles_df["section"] = section_name.replace("_", " ")

    # rename columns
    articles_df = articles_df.rename(columns = KEYS_MAPPING)

    articles_df = articles_df[[
        "id",
        "newspaper",
        "section",
        "date",
        "url",
        "title",
        "summary",
        "text",
        "file_path"
    ]]
    
    return articles_df


def process_batch_articles(articles: list, section_name: str, processed_ids: set) -> bool:
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
    articles_df = parse_articles(articles, section_name)
    original_num = articles_df.shape[0]

    # get the oldest record in the df
    min_date = datetime.strptime(articles_df.date.min(), "%Y-%m-%dT%H:%M:%S")
    # LOGGER.debug(f"min date found: {min_date}")

    if min_date.year < END_YEAR:
        end = True

       # filter df to keep only the needed articles
        articles_df = articles_df.loc[articles_df.date >= f"{END_YEAR}-01-01"]
        new_num  = articles_df.shape[0]
        LOGGER.debug(f"{original_num - new_num} articles already processed")
    else:
        end = False

    # remove articles that were already saved
    articles_df = articles_df.loc[~articles_df.id.isin(processed_ids)]

    # write results
    for file_path, group in articles_df.groupby("file_path"):
        group = group.drop(columns=["file_path"])
        articles_data = group.to_dict(orient="records")
        write_to_json_safe(articles_data, file_path)

    if articles_df.shape[0] > 0:
        # update processed ids set
        processed_ids = processed_ids.union(set(articles_df.id))

        # update file with processed ids
        save_processed_ids(NEWSPAPER_NAME, section_name, processed_ids)

    return end


def get_section_data(section_name: str):
    """
    Get all the articles from the given section
    """
    section_id = SECTIONS[section_name]

    if section_name in KEYS_NAMES:
        results_key = KEYS_NAMES[section_name]
    else:
        results_key = NEWS_KEY.format(section=section_id)

    if section_name in OP_SECTION_NAME:
        op_name = OPERATION_NAME.format(section=OP_SECTION_NAME[section_name])
    else:
        op_name = OPERATION_NAME.format(section=section_id)

    # get extra query for the section if exists
    extra_query = EXTRA_QUERY.get(section_name, "")

    # build query
    section_query = QUERY.format(
        op_name=op_name,
        key=results_key,
        section=section_id,
        extra_query=extra_query
    )

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
        params = {
            "operationName": op_name,
            "variables": {
                "where": {
                    "offsetPagination": {"size": BATCH_SIZE, "offset": offset}
                }
            },
            "query": section_query
        }
        payload = json.dumps(params)
        response = requests.post(SEARCH_URL, data=payload, headers=HEADERS)
       
        # raise exception if status != 200
        response.raise_for_status()

        response = response.json()
        if "errors" in response:
            raise Exception(f"Couldn't get data for batch {batch_num}")

        data = response["data"][results_key]

        if np.isinf(total_results):
            # update total results with real number
            total_results = data["pageInfo"]["offsetPagination"]["total"]
            LOGGER.info(f"{total_results} total results for section")

        if total_results == 0:
            LOGGER.info("No more results")
            # the last offset with results was the previous one
            offset = offset - BATCH_SIZE
            break

        # process and save articles
        articles = map(itemgetter("node"), data["edges"])
        final_page = process_batch_articles(articles, section_name, processed_ids)

        if final_page:
            LOGGER.info(f"Finished at batch {batch_num}")
            break
        else:
            save_section_checkpoint(NEWSPAPER_NAME, section_name, str(offset))
        
        # go to next batch
        offset += BATCH_SIZE

        # sleep to avoid getting blocked
        if batch_num % 20 == 0:
            sleep(random.randint(1,3))

    # save negative offset to indicate that this section is done
    save_section_checkpoint(NEWSPAPER_NAME, section_name, str(-offset))


def scrape_animal_politico():
    for section_name in SECTIONS:
        LOGGER.info(f"Getting {section_name} data")

        try:
            get_section_data(section_name)
        except Exception:
            LOGGER.error(f"Error getting data from {section_name} section", exc_info=True)


if __name__ == "__main__":
    scrape_animal_politico()