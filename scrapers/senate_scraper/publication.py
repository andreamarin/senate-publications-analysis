import os
import re
import hashlib
import logging
import requests
from time import sleep
from bs4 import BeautifulSoup
from datetime import datetime
from PyPDF2 import PdfFileReader

from utils.config import BASE_URL, BASE_URL_V2

LOGGER = logging.getLogger(__name__)


class SenatePublication():
    def __init__(self, comm_type: str, table_data, download_path: str, page: int):
        self.type = comm_type
        self._page = page
        self.__raw_table_data = str(table_data)
        self.__table_data = table_data.find_all("td")
        self.__download_path = download_path

        self.__get_date()
        self.__get_id()

    def __get_date(self):
        date_col = self.__table_data[2]
        self.date = datetime.strptime(date_col.text, "%Y/%m/%d")

    def __get_id(self):
        original_url = self.__table_data[-1].find("a")

        if original_url is None:
            self.__full_data = False
            url = self.__table_data[0].find("a").attrs["href"]

            if "https" not in url:
                url = re.sub(r"^/", "", url)
                self.url = f"{BASE_URL_V2}/{url}"
            else:
                self.url = url
        else:
            self.__full_data = True
            url = original_url.attrs["href"]

            if "https" not in url:
                url = re.sub(r"^/", "", url)
                self.url = f"{BASE_URL}/{url}"
            else:
                self.url = url

        # get id form doc's url
        hash_obj = hashlib.md5(self.url.encode("utf8"))
        self._id = hash_obj.hexdigest()

    def build_full_doc(self):
        self.__get_summary()
        self.__get_authors_data()

        if self.__full_data:
            self.__get_url_data()
            self.__validate_data()
            self.__get_full_text()
        else:
            self.full_text = self.summary

    def __get_summary(self):
        summary = self.__table_data[1].text
        self.summary = summary.replace("\n", " ")

    def __get_authors_data(self):
        """
        Get the senators and political parties involved
        """
        authors_text = self.__table_data[3].get_text(separator="\n", strip=True)

        if authors_text == "":
            LOGGER.warning(f"No authors data found for {self.url}")
            self.authors = []
            self.parties = []
        else:
            self.authors = []
            parties = set()
            for author in authors_text.split("\n"):
                author_info = re.match(r"(.+?) \((.*)\)", author)
                self.authors.append(author_info.group(1))
                parties.add(author_info.group(2))

            self.parties = list(parties)

    def __get_url_data(self):
        LOGGER.debug(self.url)

        tries = 1
        success = False
        while not success and tries <= 5:
            try:
                response = requests.get(self.url)
            except Exception:
                LOGGER.warning(f"Error loading url {self.url}, retrying ({tries})...")
                sleep(tries*2)
                tries += 1                
            else:
                if response.text == "Connection failed: Too many connections":
                    LOGGER.warning(f"Too many connections error, retrying ({tries})...")
                    sleep(tries*2)
                    tries += 1  
                else:
                    success = True
        
        if not success:
            raise Exception("Couldnt load url")
            
        self.__bs = BeautifulSoup(response.text, "lxml")

    def __validate_data(self):
        """
        Check if the loaded page has a redirect url, if so use that as the 
        publication's url and load it
        """
        script_data = self.__bs.find("script")

        if "window.location.href" in script_data.text:
            # get the real url for the publication
            new_url = re.search(r"window\.location\.href = \"(.*)\"", script_data.text).group(1)
            new_url = new_url.replace("http", "https")

            # replace for the real url
            self.url = new_url
            self.__get_url_data()

    def __download_and_parse_doc(self):
        """
        Download the doc and get the full text from the pdf
        """
        doc_name = self.doc_url.split("/")[-1]
        self.doc_path = f"{self.__download_path}/{doc_name}"

        # download doc
        response = requests.get(self.doc_url)
        if response.status_code != 200:
            LOGGER.warning(f"Couldn't download file {self.doc_url} , status {response.status_code}")
            self.full_text = self.summary
        else:
            with open(self.doc_path, "wb") as f:
                f.write(response.content)

            # get text from pdf
            self.__get_pdf_text()
        
    def __get_full_text(self):
        main_container = self.__bs.find("div", {"class": "container-fluid bg-content main"})

        if main_container is None:
            self.__get_full_text_v2()
        else:
            panel = main_container.find("div", {"class": "panel-group"}).find_all("div", {"class": "panel panel-default"}, recursive=False)[2]

            heading = panel.find("div", "panel-heading")

            if heading is not None and "Archivos para descargar" in heading.text: 
                # there is a doc to download
                self.doc_url = panel.find("a").attrs["href"]
                
                self.__download_and_parse_doc()
            else:
                self.full_text = panel.get_text(separator="\n", strip=True)

    def __get_full_text_v2(self):
        main_container = self.__bs.find("div", {"class": "container-fluid main"})

        if main_container is None:
            LOGGER.warning(f"No data for {self.url}")
            self.full_text = self.summary
            return
        
        # get all the headers in the main container
        header_divs = main_container.find_all("div", {"class": "card-header"})
        headers = [h.text.strip() for h in header_divs]

        try:
            header_pos = headers.index("Archivos para descargar:")
        except ValueError:
            LOGGER.debug("Download doc not found")
            doc_panel = None
        else:
            # get the body after the download header
            doc_panel = header_divs[header_pos].find_next_sibling(attrs={"class": "card-body"})

        if doc_panel is not None: 
            # there is a doc to download
            self.doc_url = doc_panel.find("a").attrs["href"]
            
            self.__download_and_parse_doc()
        else:
            text_panel = main_container.find_all("div", {"class": "card-body"})[1]
            self.full_text = text_panel.get_text(separator="\n", strip=True)

    def __get_pdf_text(self):
        try:           
            pdf = PdfFileReader(open(self.doc_path, "rb"))
            LOGGER.debug(f"pdf has {pdf.numPages} pages")

        except Exception:
            pdf = PdfFileReader(open(self.doc_path, "rb"), strict=False)
            LOGGER.debug(f"pdf has {pdf.numPages} pages")

        pages_texts = []
        for page_num in range(pdf.numPages):
            LOGGER.debug(f"Getting text from page {page_num}")
            page = pdf.getPage(page_num)
            page_text = page.extractText()

            # clean text
            page_text = page_text.strip()
            page_text = re.sub(r"(\n *)+", "\n", page_text)

            pages_texts.append(page_text)

        self.full_text = "\n".join(pages_texts)

    def get_json(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") or  k == "_id"}
    
    def save_table_data(self):

        # build doc name
        doc_path = self.__download_path.replace("downloads", "errors")
        doc_id = self.url.split("/")[-1]
        doc_name = f"{doc_path}/{self.type}_{doc_id}.html"

        if not os.path.exists(doc_path):
            os.makedirs(doc_path)

        with open(doc_name, "w") as f:
            f.write(self.__raw_table_data)
