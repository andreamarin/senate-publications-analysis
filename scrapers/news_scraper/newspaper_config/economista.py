NEWSPAPER_NAME = "el_economista"

# search config
SEARCH_URL = "https://www.eleconomista.com.mx/endpoints/3.0/news-list-section.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
}
BATCH_SIZE = 40

# sections
SECTIONS = [
    "sectorfinanciero",
    "empresas",
    "mercados",
    "economia",
    "estados",
    "politica",
    "opinion",
    "finanzaspersonales",
    "internacionales",
    "arteseideas",
    "tecnologia",
    "deportes",
    "autos",
    "capital-humano",
    "el-empresario",
    "econohabitat"
]


# processing vars
NUM_THREADS = 8
RENAME_COLUMNS = {
    "main.title.article": "title",
    "info.section.name": "section",
    "info.link.canonical": "url",
    "info.date.created": "date"
}