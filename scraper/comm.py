import re
import requests
from bs4 import BeautifulSoup

from config import BASE_URL

class SenateComm():
    def __init__(self, comm_type: str, day: str, href: str):
        self.type = comm_type
        self.day = day
        self.url = f"{BASE_URL}/{href}"

        self.__get_source()
        self.__build_comm()

    def __get_source(self):
        response = requests.get(self.url)
        self.__bs = BeautifulSoup(response.text, "lxml")

    def __build_comm(self):
        self.__get_main_text()
        self.__get_summary()
        self.__get_authors()
        self.__get_party()
        self.__get_full_text()

    def __get_main_text(self):
        self.__main_container = self.__bs.find_all("div", {"class": "container-fluid bg-content main"})
        main_panel = self.__main_container.find_all("div", {"class": "panel panel-default"})[1]
        panel_info = main_panel.find("div", {"class": "panel-body"}).find_all("p")

        self.__main_text = panel_info[0].text
        self.__comissions_text = panel_info[1].text

    def __get_summary(self):
        summary = self.__bs.find("div", {"id": "sinopsis1"})
        if summary is not None:
            self.summary = summary.text
        elif self.type.lower() == "iniciativas":
            summary_start = re.search("con proyecto de decreto por el que", self.__main_text).end()
            self.summary = "Propone que se " + self.__main_text[summary_start + 1:]
        elif self.type.lower() == "proposiciones":
            summary_start = re.search("con punto de acuerdo por el que se ", self.__main_text).end()
            self.summary = "Se " + self.__main_text[summary_start + 1:]

    def __get_authors(self):
        senator_regex = [
            re.compile(r"[dD]el Sen(?:\.|ador)* (.*?)(?:,| y)"),
            re.compile(r"[dD]e la Sen(?:\.|adora)* (.*?)(?:,| y)"),
            re.compile(r"[dD]e l[ao]s senador[ae]s (.* y .*?)(?:,| y)"),
            re.compile(r"De las senadoras y de los senadores del (Grupo Parlamentario(?: del*)* .+?),")
        ]

        self.authors = []
        for regex in senator_regex:
            senators = regex.finditer(self.__main_text)
            for senator in senators:
                senator_names = senator.group(1).split(" y ")
                self.authors.extend(senator_names)

    def __get_party(self):
        party_regex = re.compile(r"del Grupo Parlamentario(?: del*)* (.+),")
        
        self.party = None
        self.party_short = None
        party = party_regex.search(self.__main_text)
        if party is not None:
            self.party = party.group(1)

            if len(self.party.split(" ")) == 1:
                self.party_short = self.party
            else:
                self.party_short = "".join(x for x in self.party if x.isupper())

    def __get_full_text(self):
        main_panel = self.__main_container.find_all("div", {"class": "panel panel-default"})[2]

        # save url to original doc
        self.doc_url = main_panel.find("a").attrs["href"]

        response = requests.get()

