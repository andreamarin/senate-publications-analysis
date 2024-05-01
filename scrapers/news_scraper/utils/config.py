import locale

# set locale language to ES
locale.setlocale(locale.LC_TIME, "es_ES")

# global variables
OUT_PATH = "./data/news/"
IDS_PATH = "./data/ids/{newspaper}/"
CHECKPOINT_PATH = "./data/checkpoints/{newspaper}/"
LOCKS_PATH = "./data/locks/"

END_YEAR = 2018