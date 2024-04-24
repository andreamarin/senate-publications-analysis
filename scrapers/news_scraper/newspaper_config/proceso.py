BASE_URL = "https://www.proceso.com.mx/"
NEWSPAPER_NAME = "Proceso"

# search config
SEARCH_URL = "https://www.proceso.com.mx/u/plantillas/home/ajax/listadoP.asp"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Content-Type": "application/x-www-form-urlencoded",
}

# section and subsection ids
SECTIONS = {
    "nacional": 1,
    "economia": 2,
    "internacional": 3,
    "opinion": 5,
    "ciencia_tecnologia": 6,
    "salud": 26,
    "medio_ambiente": 27,
    "cultura": 7,
    "deportes": 8
}
SUBSECTIONS = {
    "salud": 6,
    "medio_ambiente": 6
}