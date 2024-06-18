import logging
import os
from datetime import date
from logging.handlers import TimedRotatingFileHandler

from config import logging as logging_config

DAYS_LOG_BACKUP = logging_config.days_log_backup
LOG_PATH = logging_config.log_folder


class Logger:

    def __init__(self, file_path: str = LOG_PATH):
        self.today = date.today()
        self.file_path = file_path
        self.__set_log()

    def __set_log(self):

        if logging_config.log_on_file:

            log_format = logging_config.format
            log_level = logging_config.log_level
            log_folder = self.file_path
            log_file_name = logging_config.log_file_name + '.' + str(os.getpid())
            log_file = log_folder + log_file_name

            # create log and set level
            logger = logging.getLogger()
            logger.setLevel(log_level)

            # create Handler
            handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=DAYS_LOG_BACKUP)

            formatter = logging.Formatter(log_format)
            handler.setFormatter(formatter)

            # finally remove all handler and add handler to logger
            for log_handler in logger.handlers[:]:  # remove all old handlers
                logger.removeHandler(log_handler)

            logger.addHandler(handler)

        else:

            logging.basicConfig(level=logging_config.log_level,
                                format=logging_config.format)
