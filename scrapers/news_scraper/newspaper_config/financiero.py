BASE_URL = "https://www.elfinanciero.com.mx"
NEWSPAPER_NAME = "el_financiero"

# search config
SEARCH_URL = "https://www.elfinanciero.com.mx/pf/api/v3/content/fetch/story-feed-sections"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
}
BATCH_SIZE = 20

BASE_PARAMS = {
    "filter": "{content_elements{_id,description{basic},text,display_date,headlines{basic},websites{elfinanciero{website_section{_id,name},website_url}}},count}",
    "_website": "elfinanciero"
}

# sections
SECTIONS = [
    "economia",
    "mercados",
    "nacional",
    "opinion",
    "estados",
    "salud",
    "transporte_y_movilidad",
    "empresas",
    "ciencia",
    "culturas",
    "tech",
    "border"
]


# processing vars
NUM_THREADS = 8
RENAME_COLUMNS = {
    "_id": "id",
    "display_date": "date",
    "description.basic":"summary",
    "headlines.basic": "title",
    "websites.elfinanciero.website_section.name": "section"
}