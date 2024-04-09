import os
import json

from utils.config import CHECKPOINT_PATH, IDS_PATH, OUT_PATH

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
        os.mkdir(year_dir)

    if os.path.isfile(file_name):
        # file exists
        with open(file_name, 'a+') as outfile:
            # go to the end of the file and remove the end ]
            outfile.seek(0, os.SEEK_END)
            outfile.seek(outfile.tell() - 1, os.SEEK_SET)
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
    
    with open(file_name, "w") as f:
        f.write(checkpoint)