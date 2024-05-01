import os
import json
import random
import requests
import logging
from time import sleep
from filelock import FileLock

from utils.config import CHECKPOINT_PATH, IDS_PATH, LOCKS_PATH, OUT_PATH

LOGGER = logging.getLogger(__name__)


def write_to_json_safe(articles_data: list, file_path: str):
    lock_path = file_path.replace(".json", ".lock")
    lock_file = os.path.join(LOCKS_PATH, lock_path)

    # create dir if not exists
    lock_dir = os.path.dirname(lock_file)
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir)

    # acquire lock
    with FileLock(lock_file, timeout=-1):
        write_to_json(articles_data, file_path)


def write_to_json(articles_data: list, file_path: str):
    """
    Write article's data into json file.
    If file exists append into the end of the file, if not create file

    Parameters
    ----------
    articles_data : list
        list with articles to write to the file
    file_path : str
        path to the json file where data will be saved
    """
    file_name = os.path.join(OUT_PATH, file_path)

    # create dir if not exists
    year_dir = os.path.dirname(file_name)
    if not os.path.exists(year_dir):
        os.makedirs(year_dir)

    if os.path.isfile(file_name) and os.path.getsize(file_name) > 0:
        # file exists
        with open(file_name, 'a+') as outfile:
            # go to the end of the file
            outfile.seek(0, os.SEEK_END)
            end_position = outfile.tell()

            # remove the ending "]"
            outfile.seek(end_position - 1, os.SEEK_SET)
            outfile.truncate()

            for article in articles_data:
                outfile.write(',')
                json.dump(article, outfile)
            
            # rewrite ending bracket
            outfile.write(']')
    else: 
        # create file
        with open(file_name, 'w+') as outfile:
            json.dump(articles_data, outfile)


def get_processed_ids(newspaper: str, section: str) -> set:
    newspaper_name = newspaper.lower()
    file_name = os.path.join(IDS_PATH.format(newspaper=newspaper_name), f"{section}.json")
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            processed_ids = set(json.load(f))
    else:
        processed_ids = set()

    return processed_ids


def save_processed_ids(newspaper: str, section: str, processed_ids: set):
    newspaper_name = newspaper.lower()
    file_name = os.path.join(IDS_PATH.format(newspaper=newspaper_name), f"{section}.json")

    # create dir if not exists
    ids_dir = os.path.dirname(file_name)
    if not os.path.exists(ids_dir):
        os.makedirs(ids_dir)
    
    with open(file_name, "w") as f:
        json.dump(list(processed_ids), f)


def get_section_checkpoint(newspaper: str, section: str) -> str:
    newspaper_name = newspaper.lower()
    file_name = os.path.join(CHECKPOINT_PATH.format(newspaper=newspaper_name), f"{section}.txt")

    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            checkpoint = f.read()
    else:
        checkpoint = None

    return checkpoint


def save_section_checkpoint(newspaper: str, section: str, checkpoint: str):
    newspaper_name = newspaper.lower()
    file_name = os.path.join(CHECKPOINT_PATH.format(newspaper=newspaper_name), f"{section}.txt")

    # create dir if not exists
    checks_dir = os.path.dirname(file_name)
    if not os.path.exists(checks_dir):
        os.makedirs(checks_dir)
    
    with open(file_name, "w") as f:
        f.write(checkpoint)


def get_url(url: str, method: str, headers: dict = None, data: str = None, params: dict = {}, max_retries: int = 3):
    num_try = 0
    response = None
    while response is None:
        try:
            if method == "GET":
                if headers is not None:
                    response = requests.get(url, params=params, headers=headers)
                else:
                    response = requests.get(url, params=params)
            
            elif method == "POST":
                if headers is not None:
                    response = requests.post(url, data=data, headers=headers)
                else:
                    response = requests.post(url, data=data)

        except Exception as ex:
            if num_try >= max_retries:
                # max retries exceeded raise error
                raise Exception(ex)
            
            retry = True
        
        else:
            if response.status_code >= 500:
                # internal server error, retry request
                if num_try >= max_retries:
                    # max retries exceeded raise error
                    response.raise_for_status()
                
                retry = True
                response = None
            else:
                retry =False
        
        finally:
            if retry:
                num_try += 1
                sleep_seconds = random.randint(1, 5)
                
                LOGGER.warning(f"Failed getting url {url}, retrying in {sleep_seconds}s...")
                sleep(sleep_seconds)

    return response