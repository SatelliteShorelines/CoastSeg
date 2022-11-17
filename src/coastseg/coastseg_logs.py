import os
from datetime import datetime
import logging


def prepare_logging():
    if not os.path.exists(os.path.abspath(os.path.join(os.getcwd(), "logs"))):
        os.mkdir(os.path.abspath(os.path.join(os.getcwd(), "logs")))


def create_root_logger():
    """Creates the root logger. The root logger will write to a log file with the format
    log_<year>-<month>-<day>-<hour>-<minute>. This log file will  be written to by all the other loggers
    """
    log_filename = "log_" + datetime.now().strftime("%m-%d-%y-%I_%M_%S") + ".log"
    log_file = os.path.abspath(os.path.join(os.getcwd(), "logs", log_filename))
    # configure the logger
    log_format = "%(asctime)s - %(filename)s at line %(lineno)s in %(funcName)s() - %(levelname)s : %(message)s"
    os.path.abspath(os.path.join(os.getcwd(), "logs"))
    # Use FileHandler() to log to a file
    file_handler = logging.FileHandler(log_file, mode="a")
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    # Have all loggers write to the same log file
    logging.basicConfig(
        handlers=[file_handler],
        format=log_format,
        level=logging.INFO,
        datefmt="-%m-%d-%y-%I:%M:%S",
    )


# Prepare and create the logger
prepare_logging()
create_root_logger()
