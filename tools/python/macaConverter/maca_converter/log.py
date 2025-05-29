import logging
import os

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

log_level = os.getenv('CONVERTER_LOG_LEVEL')
valid_log_level = ['10', '20', '30', '40', '50']

def getLogger(name, level):
    #print('get log_level:', log_level)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()

    if log_level in valid_log_level:
        console_handler.setLevel(int(log_level))
    else:
        console_handler.setLevel(level)
            
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

