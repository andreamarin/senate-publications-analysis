import sys
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta

# local imports
import utils
from config import BASE_URL, MAIN_URL

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)


def process_data(data: BeautifulSoup, section_type: str, day: str) -> dict:
    """
    Process an initiative/proposal data and return formatted dict

    Parameters
    ----------
    data : BeautifulSoup
        bs4 object with the comm's html
    section_type : str
        indicates if it's a proposal or an initiative
    day : str
        day the proposal was published

    Returns
    -------
    dict
        dictionary with the proposal's complete information
    """
    document_href = data.find("a")["href"]
    LOGGER.info(document_href)


def process_day(day: str, href: str):
    url = f"{BASE_URL}/{href}"

    response = requests.get(url)
    day_source = response.text

    LOGGER.info(f"Processing {day}")
    relevant_sections = utils.initiatives_proposals_found(day_source)

    if len(relevant_sections) == 0:
        LOGGER.info(f"No initiatives or proposals in {href}")

    for section_type, name in relevant_sections.items():
        LOGGER.info(f"{section_type}...")

        for data in utils.get_section_data(name, day_source):
            process_data(section_type, data)


def senate_scraper(start_date: datetime, end_date: datetime):
    # load main page
    LOGGER.info("Loading main page")
    response = requests.get(MAIN_URL)
    main_source = response.text

    start_month = start_date.replace(day=1)
    end_month = end_date.replace(day = 1)

    current_month = start_month
    while current_month <= end_month:

        # get proper generator depending on the current month
        if current_month == start_month:
            month_urls = utils.get_month_urls(current_month, main_source, start_date=start_date)
        elif current_month == end_month:
            month_urls = utils.get_month_urls(current_month, main_source, end_date=end_date)
        else:
            month_urls = utils.get_month_urls(current_month, main_source)
    
        for day, href in month_urls:
            process_day(day, href)

        current_month += relativedelta(months=1)


if __name__ == "__main__":
    start_date = utils.parse_date(sys.argv[1])
    end_date = utils.parse_date(sys.argv[2])

    LOGGER.info(f"processing from {start_date} to {end_date}")

    senate_scraper(start_date, end_date)