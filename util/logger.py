import logging
import os


class Logger:
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # create console handler with a specific level
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # create file handler which logs even debug messages
        if not os.path.exists('log'):
            os.makedirs('log')
        fh = logging.FileHandler('log/app.log')
        fh.setLevel(level)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)


logger = Logger(__name__).logger
