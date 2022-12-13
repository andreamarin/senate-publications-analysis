import os
import logging
import requests

# local imports
import utils
from chrome_driver import ChromeDriver
from config import DRIVER_PATH, MAIN_URL

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)


def process_month(month, page_source, driver):
    days = utils.get_month_urls(month, page_source)


def senate_scraper(start_date, end_date):
    download_path = os.path.join(os.getcwd(), "data")
    LOGGER.info(f"Create driver with download path: {download_path}")
    driver = ChromeDriver(driver_path=DRIVER_PATH, download_path=download_path, headless=False)

    # load main page
    LOGGER.info("Loading main page")
    driver.get(MAIN_URL)

    main_source = driver.page_source


if __name__ == "__main__":
    senate_scraper()