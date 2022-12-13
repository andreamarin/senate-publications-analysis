import os
from selenium.webdriver.common.by import By
from selenium.webdriver import Chrome, ChromeOptions

class ChromeDriver(Chrome):
    def __init__(self, driver_path: str = None, headless: bool = True, download_path: str = None):
        if driver_path is None:
            self.driver_path = "/usr/local/bin/"
        else:
            self.driver_path = driver_path

        if download_path is None:
            self.download_path = os.path.join(os.getcwd(), "/downloads")
        else:
            self.download_path = download_path

        self.headless = headless
        
        # build chrome driver
        self._build_driver()

    def _build_driver(self):
        """
        build chrome driver with the appropriate options
        """
        options = ChromeOptions()

        if self.headless:
            options.set_headless(True)

        # set download path
        if not os.path.exists(self.download_path):
            os.mkdir(self.download_path)
        prefs = {
            "download.default_directory": self.download_path,
            "download.prompt_for_download": False,
            "plugins.plugins_disabled": ["Chrome PDF Viewer"],
            "download.directory_upgrade": True
        }
        options.add_experimental_option("prefs", prefs)

        # set automatic downloads
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-print-preview")

        # start window at max size
        options.add_argument("--start-maximized")

        # init driver
        super().__init__(executable_path=self.driver_path, chrome_options=options)

    def open_in_new_window(self, url: str):
        # Open a new window
        self.execute_script("window.open('');")
        
        # switch to the new window
        self.switch_to.window(self.window_handles[1])

        # open url
        self.get(url)