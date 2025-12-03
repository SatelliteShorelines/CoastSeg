import os
from datetime import datetime
import logging
from coastseg import core_utilities
# DESCRIPTION: Sets up a logging system that writes logs to a file named with the current timestamp in a "logs" directory.


def prepare_logging():
    """Create the logs/ directory if it doesn't exist.

    The directory is created under the CoastSeg project base directory as
    determined by :func:`core_utilities.get_base_dir`.
    """
    if not os.path.exists(os.path.abspath(os.path.join(core_utilities.get_base_dir(), "logs"))):
        os.mkdir(os.path.abspath(os.path.join(core_utilities.get_base_dir(), "logs")))


def clear_default_handlers() -> None:
    """Removes the default logging handlers"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def create_root_logger()-> None:
    """Configure the root logger to write to a timestamped file in logs/.

    The log filename uses the pattern:
    log_<MM-DD-YY>-<HH>_<MM>_<SS>.txt (12-hour clock with seconds), and is
    placed in <base_dir>/logs. The log format includes timestamp, filename,
    line number, function name, level, and the message.

    Returns:
        None

    Side Effects:
        - Affects all loggers (since it configures the root logger).
    """
    log_filename = "log_" + datetime.now().strftime("%m-%d-%y-%I_%M_%S") + ".txt"
    log_file = os.path.abspath(os.path.join(core_utilities.get_base_dir(), "logs", log_filename))
    # configure the logger
    log_format = "%(asctime)s - %(filename)s at line %(lineno)s in %(funcName)s() - %(levelname)s : %(message)s"
    os.path.abspath(os.path.join(core_utilities.get_base_dir(), "logs"))
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
# clear the default logging handlers this is needed for the logs to work in google colab
clear_default_handlers()
create_root_logger()
