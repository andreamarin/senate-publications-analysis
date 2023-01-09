
import re
import time
import logging
from datetime import datetime
from bs4 import BeautifulSoup

from .config import TABLE_XPATH

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


def wait_new_page(driver, new_page: int):
    loaded_page = False

    while not loaded_page:
        main_table = driver.get_element(TABLE_XPATH)
        page_source = main_table.get_attribute("outerHTML")

        # get current page
        bs = BeautifulSoup(page_source,'lxml')
        selected_page = bs.find("ul", {"class": "pagination"}).find("li", {"class": "active"})

        LOGGER.debug(f"Current page: {selected_page}")

        if int(selected_page.text) == new_page:
            loaded_page = True
        else:
            time.sleep(5)

    return main_table
