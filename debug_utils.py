import sys, inspect, pprint, logging, os

# #################################  Log  ################################# #
DEBUG = 0
INFO = 1
WARNING = 2
ERROR = 3
CRITICAL = 4

LOGGER_NAME = 'serv_rate'

logger = logging.getLogger(LOGGER_NAME)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

FORMAT = '%(levelname)s] %(func_name)s: %(msg)s'
formatter = logging.Formatter(FORMAT)

def log_to_std():
  logger = logging.getLogger(LOGGER_NAME)
  sh = logging.StreamHandler()
  sh.setFormatter(formatter)
  logger.addHandler(sh)

level_log_m = {INFO: logger.info, DEBUG: logger.debug, WARNING: logger.warning, ERROR: logger.error, CRITICAL: logger.critical}

def log_to_file(filename, directory=None):
  if directory and not os.path.exists(directory):
    os.makedirs(directory)

  logger = logging.getLogger(LOGGER_NAME)

  filepath = '{}/{}'.format(directory, filename)
  fh = logging.FileHandler(filepath, mode='w')
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

def get_extra():
  frame = inspect.currentframe().f_back.f_back.f_code
  return {'func_name': '{}::{}'.format(os.path.split(frame.co_filename)[1], frame.co_name)}

def log(level: int, _msg_: str, **kwargs):
  level_log_m[level]("{}\n{}".format(_msg_, pstr(**kwargs)), extra=get_extra())

## Always log
def alog(level: int, _msg_: str, **kwargs):
  logger.critical("{}\n{}".format(_msg_, pstr(**kwargs)), extra=get_extra())

def pstr(**kwargs):
  s = ''
  for k, v in kwargs.items():
    s += "  {}: {}\n".format(k, pprint.pformat(v))
  return s

# ###############################  Assert  ############################### #
def check(condition: bool, _msg_: str, **kwargs):
  if not condition:
    logger.error("{}\n{}".format(_msg_, pstr(**kwargs)), extra=get_extra())
    raise AssertionError()

def assert_(_msg_: str, **kwargs):
  logger.error("{}\n{}".format(_msg_, pstr(**kwargs)), extra=get_extra())
  raise AssertionError()
