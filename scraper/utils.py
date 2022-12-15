import re
import logging
from datetime import datetime
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)


def parse_date(date_str: str) -> datetime:
    """
    Parse a yyyy-mm-dd date string to datetime
    Returns error if the date is not formatted properly

    Parameters
    ----------
    date_str : str
        string representing the date

    Returns
    -------
    datetime
    """

    assert re.match(r"\d{4}-\d{2}-\d{2}", date_str) is not None, "Date must be in yyyy-mm-dd format"
    date = datetime.strptime(date_str, "%Y-%m-%d")

    return date


def get_total_pages(page_source: str):
    bs = BeautifulSoup(page_source, "lxml")
    pages_info = bs.find("div", {"class": "panel-heading"}).find("p")

    # use regex to find total
    total_pages = re.search(r"Página \d+ de (\d+),", pages_info.text).group(1)

    return int(total_pages)


def get_page_comms(page_source: str):
    bs = BeautifulSoup(page_source, "lxml")
    comms_table = bs.find("table").find("tbody")
    return comms_table.find_all("tr")