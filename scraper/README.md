# Senata bot

This folder contains the code for the bot that gets the mexican Senate's initiatives from the webpage.

## Installation

To able to use selenium you need to download the [chromedriver](https://sites.google.com/chromium.org/driver/downloads) that corresponds to the Chrome version you are using.

Save the chomedriver in the `/usr/local/bin` path.

The python libraries needed

## How to use

The bot gets all the initiatives and proposals from the mexican senate created within the given dates. For this the script receives 2 parameters:
* `start_date`: first day to get the publications from (inclusive)
* `end_date`: last day to get the publicatiosn from (inclusive)

> Both dates must be in yyyy-mm-dd format

**Example:**
```bash
python3 senate_bot.py 2022-01-01 2022-07-25
```

The publications are taken from:

* initiatives – https://pleno.senado.gob.mx/infosen/infosen64/index.php?c=Legislatura65&a=iniciativas
* porposals – https://pleno.senado.gob.mx/infosen/infosen64/index.php?c=Legislatura65&a=proposiciones

