import sys
import colorlog
import logging


def create_logger(
    name,
    filename=None,
    path="./log",
    level=colorlog.INFO,
    log_str="%(levelname)s - %(name)s - %(message)s",
):
    log = colorlog.getLogger()
    log.level = level

    file_log_str = f"%(asctime)s: {log_str}" if "asctime" not in log_str else log_str
    color_log_str = f"%(log_color)s{log_str}" if "log_color" not in log_str else log_str

    fileHandler = logging.FileHandler("{0}/{1}.log".format(path, filename or name))
    fileHandler.setFormatter(logging.Formatter(file_log_str))
    log.addHandler(fileHandler)

    consoleHandler = colorlog.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(colorlog.ColoredFormatter(color_log_str))
    log.addHandler(consoleHandler)

    return log


def set_level(log, level):
    if level == "DEBUG":
        log.setLevel(logging.DEBUG)
    elif level == "INFO":
        log.setLevel(logging.INFO)
    elif level == "WARNING":
        log.setLevel(logging.WARNING)
    elif level == "ERROR":
        log.setLevel(logging.ERROR)
    elif level == "CRITICAL":
        log.setLevel(logging.CRITICAL)


def get_logger(name):
    log = colorlog.getLogger(name)

    return log
