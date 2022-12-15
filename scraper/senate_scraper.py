import os
import sys
import json
import logging
from datetime import datetime

# local imports
import utils
from chrome_driver import ChromeDriver
from publication import SenatePublication
from config import *


# setup loggers
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
critical_logs = ["urllib3", "selenium"]
for logger_name in critical_logs:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def process_comms(full_comms):
    for i, comm in enumerate(full_comms):
        if i % 20 == 0:
            LOGGER.info(f"Saved {i} {comm.type}")

        comm.build_full_doc()

        date_path = comm.date.strftime("year=%Y/month=%m/day=%d")
        save_path = f"{os.getcwd()}/{comm.type}/{date_path}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        json_path = f"{save_path}/{comm.id}.json"
        with open(json_path, "w") as f:
            comm_dict = {k: v for k, v in comm.__dict__.items() if not k.startswith("_")}
            comm_dict["date"] = comm_dict["date"].isoformat()
            json.dump(comm_dict, f)


def process_page(page_source, start_date, end_date, comm_type):
    page_comms = []
    for data in utils.get_page_comms(page_source):
        comm = SenatePublication(comm_type, data, DOWNLOAD_PATH)

        if comm.date >= start_date and comm.date <= end_date:
            page_comms.append(comm)

    return page_comms


def load_legislature_data(legis_number: str, comm_type: str, start_date, end_date):

    # build url
    LOGGER.info(f"Loading {comm_type} from the {legis_number}th legislature")
    url = MAIN_URL.format(legis_number=legis_number, type=comm_type)

    driver = ChromeDriver(driver_path=DRIVER_PATH, headless=HEADLESS, download_path=DOWNLOAD_PATH)
    driver.get(url)

    main_table = driver.get_element(TABLE_XPATH)

    current_page = 1
    total_pages = utils.get_total_pages(main_table.get_attribute("outerHTML"))

    full_comms = []
    while current_page <= total_pages:
        LOGGER.info(f"Processing page {current_page} out of {total_pages}")
        page_comms = process_page(main_table.get_attribute("outerHTML"), start_date, end_date, comm_type)
        full_comms.extend(page_comms)

        current_page += 1
        if current_page <= total_pages:
            driver.execute_script(LOAD_PAGE_SCRIPT.format(page_num=current_page))
            driver.wait_elemet_is_old(main_table)

            # get new main table
            main_table = driver.get_element(TABLE_XPATH)


    LOGGER.info(f"Processing {len(full_comms)} {comm_type}")
    process_comms(full_comms)
    LOGGER.info(f"Finished saving {comm_type}")

    driver.close()


def senate_scraper(start_date: datetime, end_date: datetime):
   
    for legis_number in LEGISLATURE_DATES:
        if start_date <= LEGISLATURE_DATES[legis_number]["end_date"] and start_date >= LEGISLATURE_DATES[legis_number]["start_date"]:
            # save data from this legislature
            legis_start = max(start_date, LEGISLATURE_DATES[legis_number]["start_date"])
            legis_end = min(end_date, LEGISLATURE_DATES[legis_number]["end_date"])

            load_legislature_data(legis_number, "iniciativas", legis_start, legis_end)
            load_legislature_data(legis_number, "proposiciones", legis_start, legis_end)


if __name__ == "__main__":
    start_date = utils.parse_date(sys.argv[1])
    end_date = utils.parse_date(sys.argv[2])

    LOGGER.info(f"processing from {start_date} to {end_date}")

    senate_scraper(start_date, end_date)