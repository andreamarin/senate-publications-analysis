import os
import re
import locale
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
    date = datetime.strptime(date_str, "%Y-%m-%d").date()

    return date


def get_month_urls(month: datetime, page_source: BeautifulSoup, start_date: datetime = None, end_date: datetime = None):
    """
    Get all the urls for the days where there was a senate meeting

    Parameters
    ----------
    month : datetime
        month to get the days from
    page_source : BeautifulSoup
        html of the main source
    start_date : datetime, optional
        first day to start getting the urls for if left as None it will get the whole month, by default None
    end_date : datetime, optional
        last day to start getting the urls for if left as None it will get the whole month, by default None

    Yields
    ------
    str, str
        date and href of the date's information
    """
    if os.name == "posix":
        locale.setlocale(locale.LC_TIME, "es_ES")
    else:
        locale.setlocale(locale.LC_ALL, "es_ES")

    month_str = month.strftime("%B %Y").capitalize()
    LOGGER.info(f"Getting days for {month_str}")

    bs = BeautifulSoup(page_source, "lxml")

    month_name = bs.find(name="p", attrs={"class": "text-center"}, text=month_str)
    for parent in month_name.parents:
        if "panel-default" in parent.attrs["class"]:
            LOGGER.debug("found parent")
            break

    days = parent.find("tbody")

    for day_info in days.find_all("a"):
        url = day_info.attrs["href"]
        LOGGER.debug(url)

        # get date from url
        url_date = re.match(r"/\d{2}/gaceta_del_senado/(\d{4}_\d{2}_\d{2})/\d+", url).group(1)
        date = datetime.strptime(url_date, "%Y_%m_%d")
        if start_date is not None and date < start_date:
            continue

        if end_date is not None and date > end_date:
            break

        date_str = date.strftime("%Y-%m-%d")
        clean_url = re.sub(r"^/", "", url)

        yield date_str, clean_url


def initiatives_proposals_found(page_source: BeautifulSoup) -> dict:
    bs = BeautifulSoup(page_source, "lxml")
    
    comms_summary = bs.find("div", {"class": "panel panel-default"})
    
    # get the type of comms that were discussed that day
    comms_types = comms_summary.find_all("li")
    if len(comms_types) == 0:
        comms_types = comms_summary.find_all("tr")

    # get names and href of all comms
    comms_info = {
        re.sub(r" \d+$", "", c.find("strong").text): c.find("a").attrs["href"]
        for c in comms_types
    }

    comms_regex = re.compile(r"Iniciativa*|Proposicion*")
    relevant_comms = {c: comms_info[c] for c in filter(comms_regex.match, comms_info.keys())}

    return relevant_comms


def get_section_data(name: str, page_source: BeautifulSoup):
    bs = BeautifulSoup(page_source, "lxml")

    # clean name
    name = name.replace("#", "")

    # find where the section starts
    section_header = bs.find("a", {"class": "ancla", "name": name})
    for comm in section_header.parents:
        if "panel-default" in comm.attrs["class"]:
            LOGGER.debug("found parent")
            break

    is_new_section = False
    while not is_new_section:
        data = comm.find("div", {"class": "panel-body"})

        table_data = data.find("td", {"style": "padding: 15px;"})
        if table_data is not None:
            data = table_data

        yield data

        # find next section
        comm = comm.find_next_sibling()
        
        # if there's a heading it's a new section
        if comm.find("div", {"class": "panel-heading"}) is not None:
            is_new_section = True
    