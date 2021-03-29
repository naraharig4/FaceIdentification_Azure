import cv2
import datetime
from datetime import datetime
import logging

formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%d-%b-%y %H:%M:%S')

class Utility:
    def __init__(self, store_path="data/images/"):
        self.storeLocation = store_path

    def saveFrametoLocal(self, frame):
        name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        name += ".jpg"
        cv2.imwrite(self.storeLocation+name, frame)

    def setup_logger(name, log_file, level=logging.INFO):
        """To setup as many loggers as you want"""

        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger

