import os
from datetime import datetime

DRIVER_PATH = "/usr/local/bin/chromedriver"
MAIN_URL = "https://pleno.senado.gob.mx/infosen/infosen64/index.php?c=Legislatura{legis_number}&a={type}"
BASE_URL = "https://www.senado.gob.mx"

HEADLESS = True
DOWNLOAD_PATH = os.path.join(os.getcwd(), "downloads")

TABLE_XPATH = '//*[@id="viewDataBase"]/div'

LOAD_PAGE_SCRIPT = 'loadData({page_num}, "fecha_presentacion", "DESC", 250, "asunto,sintesis,fechaPresentacion,autores,turno,leyesModifica,aprobacion,estatus,camaraOrigen,resolutivo")'

LEGISLATURE_DATES = {
    "64": {
        "start_date": datetime(2018, 9, 1),
        "end_date": datetime(2021,8 ,31)
    },
    "65": {
        "start_date": datetime(2021, 9, 1),
        "end_date": datetime(2024,8 ,31)
    }
}

