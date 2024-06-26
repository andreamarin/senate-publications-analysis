import sys
import logging
from datetime import datetime, timedelta

# local imports
from chrome_driver import ChromeDriver
from publication import SenatePublication
from utils import methods
from utils.config import *
from utils.db import connect_mongo_db, save_publications, publication_exists


# setup loggers
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
critical_logs = ["urllib3", "selenium", "PyPDF2"]
for logger_name in critical_logs:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def process_comms(full_comms: list, conn):
    """
    Finish processing all the publications

    Parameters
    ----------
    full_comms : list
        list with all the publication objects to process
    conn :
        client to the Mongo DB
    """

    for i, comm in enumerate(full_comms):
        if i % 20 == 0:
            LOGGER.info(f"Saved {i} {comm.type}")

        # get the full data
        if not publication_exists(comm._id, TABLE_NAME, conn):
            try:
                comm.build_full_doc()
            except Exception:
                LOGGER.error(f"Couldn't process publication {comm.url}, from page {comm._page}", exc_info=True)
                comm.save_table_data()
            else:
                save_publications(comm.get_json(), TABLE_NAME, conn)
        else:
            LOGGER.debug(f"Publication {comm._id} has already been processed")


def process_page(page_source: str, start_date: datetime, end_date: datetime, comm_type: str, page_num: int, conn) -> list:
    """
    Get all the relevant publications from the current page

    Parameters
    ----------
    page_source : str
        html of the current page
    start_date : datetime
        lower bound for the dates fo the files to process
    end_date : datetime
        upper bound for the dates fo the files to process
    comm_type : str
        iniciativas or proposiciones
    page_num : int
        number of the page to process
    
    Returns
    -------
    list
        list with the publications that need to be processed
    """
    page_comms = []

    out_of_range = total_comms = processed_comms = 0
    for data in methods.get_page_comms(page_source):
        comm = SenatePublication(comm_type, data, DOWNLOAD_PATH, page_num)

        if comm.date >= start_date and comm.date <= end_date:

            if not publication_exists(comm._id, TABLE_NAME, conn):
                page_comms.append(comm)
            else:
                LOGGER.debug(f"Publication {comm._id} has already been processed")
                processed_comms += 1
        
        else:
            out_of_range += 1

        total_comms += 1

    LOGGER.info(f"{len(page_comms)} out of {total_comms} publications to process")
    LOGGER.debug(f"{processed_comms} are already processed")
    LOGGER.debug(f"{out_of_range} are out of the provided date range")

    return page_comms


def load_legislature_data(legis_number: str, comm_type: str, start_date: datetime, end_date: datetime, conn):
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

    # get total pages to process
    main_table = driver.get_element(TABLE_XPATH)
    total_pages = methods.get_total_pages(main_table.get_attribute("outerHTML"))

    full_comms = []
    for current_page in range(1, total_pages+1):
        # load page in the desired order
        loaded_page = methods.wait_new_page(driver, current_page, main_table)

        if not loaded_page:
            LOGGER.error(f"Can't load page number {current_page}")
            break

        # get new main table
        main_table = driver.get_element(TABLE_XPATH)

        LOGGER.info(f"Processing page {current_page} out of {total_pages}")
        page_comms = process_page(main_table.get_attribute("outerHTML"), start_date, end_date, comm_type, current_page, conn)
        full_comms.extend(page_comms)

    driver.close()
            
    LOGGER.info(f"Processing {len(full_comms)} {comm_type}")
    process_comms(full_comms, conn)
    LOGGER.info(f"Finished saving {comm_type}")


def senate_bot(start_date: datetime, end_date: datetime):

    conn = connect_mongo_db(DB_NAME)
   
    for legis_number in LEGISLATURE_DATES:
        if start_date <= LEGISLATURE_DATES[legis_number]["end_date"] and start_date >= LEGISLATURE_DATES[legis_number]["start_date"]:
            # save data from this legislature
            legis_start = max(start_date, LEGISLATURE_DATES[legis_number]["start_date"])
            legis_end = min(end_date, LEGISLATURE_DATES[legis_number]["end_date"])
            
            try:
                load_legislature_data(legis_number, "iniciativas", legis_start, legis_end, conn)
            except Exception:
                LOGGER.error("Error loading iniciativas", exc_info=True)

            try:
                load_legislature_data(legis_number, "proposiciones", legis_start, legis_end, conn)
            except Exception:
                LOGGER.error("Error loading proposiciones", exc_info=True)

            start_date = LEGISLATURE_DATES[legis_number]["end_date"] + timedelta(days=1)

if __name__ == "__main__":
    start_date = methods.parse_date(sys.argv[1])
    end_date = methods.parse_date(sys.argv[2])

    LOGGER.info(f"processing from {start_date} to {end_date}")

    senate_bot(start_date, end_date)