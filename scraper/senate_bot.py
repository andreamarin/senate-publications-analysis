import sys
import logging
from datetime import datetime

# local imports
from chrome_driver import ChromeDriver
from publication import SenatePublication
from utils import methods
from utils.config import *
from utils.db import save_publication, publication_exists


# setup loggers
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
critical_logs = ["urllib3", "selenium", "PyPDF2"]
for logger_name in critical_logs:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def process_comms(full_comms: list):
    """
    Finish processing all the publications

    Parameters
    ----------
    full_comms : list
        list with all the publication objects to process
    """
    for i, comm in enumerate(full_comms):
        if i % 20 == 0:
            LOGGER.info(f"Saved {i} {comm.type}")

        # get the full data
        try:
            comm.build_full_doc()
        except Exception:
            LOGGER.error(f"Couldn't process publication {comm.id}", exc_info=True)
        else:
            save_publication(comm)


def process_page(page_source, start_date, end_date, comm_type):
    """
    Get all the relevant publications from the current page

    Parameters
    ----------
    page_source : _type_
        _description_
    start_date : _type_
        _description_
    end_date : _type_
        _description_
    comm_type : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    page_comms = []
    for data in methods.get_page_comms(page_source):
        comm = SenatePublication(comm_type, data, DOWNLOAD_PATH)

        if comm.date >= start_date and comm.date <= end_date:

            if not publication_exists(comm.id, comm.date):
                page_comms.append(comm)
            else:
                LOGGER.info(f"Publication {comm.id} has already been processed")

    return page_comms


def load_legislature_data(legis_number: str, comm_type: str, start_date, end_date):
    """
    Load data from the given legislatura

    Parameters
    ----------
    legis_number : str
        legislature number to process
    comm_type : str
        type of publication (iniciativas or proposiciones)
    start_date : datetime
        first date to get publications from
    end_date : datetime
        last date to get publications from
    """
    # build url
    LOGGER.info(f"Loading {comm_type} from the {legis_number}th legislature")
    url = MAIN_URL.format(legis_number=legis_number, type=comm_type)

    driver = ChromeDriver(driver_path=DRIVER_PATH, headless=HEADLESS, download_path=DOWNLOAD_PATH)
    driver.get(url)

    main_table = driver.get_element(TABLE_XPATH)

    current_page = 1
    total_pages = methods.get_total_pages(main_table.get_attribute("outerHTML"))

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


def senate_bot(start_date: datetime, end_date: datetime):
   
    for legis_number in LEGISLATURE_DATES:
        if start_date <= LEGISLATURE_DATES[legis_number]["end_date"] and start_date >= LEGISLATURE_DATES[legis_number]["start_date"]:
            # save data from this legislature
            legis_start = max(start_date, LEGISLATURE_DATES[legis_number]["start_date"])
            legis_end = min(end_date, LEGISLATURE_DATES[legis_number]["end_date"])

            load_legislature_data(legis_number, "iniciativas", legis_start, legis_end)
            load_legislature_data(legis_number, "proposiciones", legis_start, legis_end)


if __name__ == "__main__":
    start_date = methods.parse_date(sys.argv[1])
    end_date = methods.parse_date(sys.argv[2])

    LOGGER.info(f"processing from {start_date} to {end_date}")

    senate_bot(start_date, end_date)