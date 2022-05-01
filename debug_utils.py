import inspect
import logging
import os
import pprint
import sys

# #################################  Log  ################################# #
DEBUG = 0
INFO = 1
WARNING = 2
ERROR = 3
CRITICAL = 4


# LOGGING_FORMAT = "%(levelname)s] %(func_name)s: %(msg)s"
LOGGING_FORMAT = "%(levelname)s:%(filename)s:%(lineno)s-%(func_name)s: %(message)s"
# LOGGING_FORMAT = "%(levelname)s:%(filename)s:%(lineno)s-%(funcName)s: %(message)s"
formatter = logging.Formatter(LOGGING_FORMAT)


LOGGER_NAME = "serv_rate"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(LOGGER_NAME)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


def log_to_std():
    logger = logging.getLogger(LOGGER_NAME)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)


level_log_m = {
    INFO: logger.info,
    DEBUG: logger.debug,
    WARNING: logger.warning,
    ERROR: logger.error,
    CRITICAL: logger.critical,
}


def log_to_file(filename, directory=None):
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    logger = logging.getLogger(LOGGER_NAME)

    filepath = "{}/{}".format(directory, filename)
    fh = logging.FileHandler(filepath, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def get_extra():
    frame = inspect.currentframe().f_back.f_back.f_code
    return {
        "func_name": "{}::{}".format(os.path.split(frame.co_filename)[1], frame.co_name)
    }


def log(level: int, _msg_: str, **kwargs):
    level_log_m[level](f"{_msg_}{pstr(**kwargs)}", extra=get_extra())


## Always log
def alog(level: int, _msg_: str, **kwargs):
    logger.critical("{}\n{}".format(_msg_, pstr(**kwargs)), extra=get_extra())


def pstr(**kwargs):
    if len(kwargs) == 0:
        return ""
    else:
        s = "\n"
        for k, v in kwargs.items():
            s += f"  {k}: {pprint.pformat(v)}\n"
        return s


# ###############################  Assert  ############################### #
def check(condition: bool, _msg_: str, **kwargs):
    if not condition:
        logger.error("{}\n{}".format(_msg_, pstr(**kwargs)), extra=get_extra())
        raise AssertionError()


def assert_(_msg_: str, **kwargs):
    logger.error("{}\n{}".format(_msg_, pstr(**kwargs)), extra=get_extra())
    raise AssertionError()
