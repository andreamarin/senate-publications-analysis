import re
import hashlib
import logging
import requests
from time import sleep
from bs4 import BeautifulSoup
from datetime import datetime

from config import BASE_URL

LOGGER = logging.getLogger(__name__)


class SenatePublication():
    def __init__(self, comm_type: str, table_data, download_path):
        self.type = comm_type
        self.__table_data = table_data.find_all("td")
        self.__download_path = download_path

        self.__get_date()

    def __get_date(self):
        date_col = self.__table_data[2]
        self.date = datetime.strptime(date_col.text, "%Y/%m/%d")

    def build_full_doc(self):
        self.__get_summary()
        self.__get_authors_data()
        self.__get_url()
        self.__build_id()
        self.__get_full_text()

    def __get_summary(self):
        summary = self.__table_data[1].text
        self.summary = summary.replace("\n", " ")

    def __get_authors_data(self):
        authors = self.__table_data[3].get_text(separator="\n", strip=True)
        
       # get all the authors and parties involves 
        self.authors = []
        parties = set()
        for author in authors.split("\n"):
            author_info = re.match(r"(.+?) \((.*)\)", author)
            self.authors.append(author_info.group(1))
            parties.add(author_info.group(2))

        self.parties = list(parties)

    def __get_url(self):
        url = self.__table_data[-1].find("a").attrs["href"]

        if "https" not in url:
            url = re.sub(r"^/", "", url)
            self.url = f"{BASE_URL}/{url}"
        else:
            self.url = url

        LOGGER.debug(url)

        tries = 1
        success = False
        while not success and tries <= 5:
            try:
                response = requests.get(self.url)
            except Exception:
                LOGGER.warning(f"Error loading url {self.url}, retrying ({tries}...)")
                sleep(tries*2)
                tries += 1                
            else:
                success = True
        
        if not success:
            raise Exception("Couldnt load url")
            
        self.__bs = BeautifulSoup(response.text, "lxml")

    def __build_id(self):
        hash_obj = hashlib.md5(self.url.encode("utf8"))
        self.id = hash_obj.hexdigest()
        
    def __get_full_text(self):
        main_container = self.__bs.find("div", {"class": "container-fluid bg-content main"})
        panel = main_container.find("div", {"class": "panel-group"}).find_all("div", {"class": "panel panel-default"}, recursive=False)[2]

        heading = panel.find("div", "panel-heading")

        if heading is not None and "Archivos para descargar" in heading.text: 
            # there is a doc to download
            self.doc_url = panel.find("a").attrs["href"]
            
            doc_name = self.doc_url.split("/")[-1]
            self.doc_path = f"{self.__download_path}/{doc_name}"

            # download doc
            response = requests.get(self.doc_url)
            with open(self.doc_path, "wb") as f:
                f.write(response.content)
        else:
            self.full_text = panel.get_text(separator="\n", strip=True)
        

