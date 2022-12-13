from datetime import datetime
import os
import re
import locale
import logging
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

def get_month_urls(month, page_source: str, start_date = None):
    if os.name == "posix":
        locale.setlocale(locale.LC_TIME, "es_ES")
    else:
        locale.setlocale(locale.LC_ALL, "es_ES")

    month_str = month.strftime("%B %Y").capitalize()
    LOGGER.info(f"Getting days for {month_str}")

    bs = BeautifulSoup(page_source)

    month_name = bs.find(name="p", attrs={"class": "text-center"}, text=month_str)
    for parent in month_name.parents:
        if "panel-default" in parent.attrs["class"]:
            LOGGER.debug("found parent")
            break

    days = parent.find("tbody")

    days_urls = {}
    for day_info in days.find_all("a"):
        url = day_info.attrs["href"]
        
        # get date from url
        url_date = re.match(r"\d{2}/gaceta_senado/(\d{4}_\d{4}_\d{4})/\d+", url).group(1)
        date = datetime.strptime(url_date, "%Y_%m_%d")
        if start_date is not None and date < start_date:
            continue

        date_str = date.strftime("%Y-%m-%d")
        if date_str in days_urls:
            days_urls[date_str].append(url)
        else:
            days_urls[date_str] = [url]

    return days_urls
