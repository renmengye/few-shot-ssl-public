# Copyright (c) 2018 Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell,
# Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, Richars S. Zemel.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
"""
A python logger

Usage:
    # Set logger verbose level.
    import os
    os.environ["VERBOSE"] = 1

    import logger
    log = logger.get("../logs/sample_log")

    log.info("Hello world!")
    log.info("Hello again!", verbose=2)
    log.warning("Something might be wrong.")
    log.error("Something is wrong.")
    log.fatal("Failed.")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import inspect
import os
import sys
import threading
import traceback

TERM_COLOR = {
    "normal": "\033[0m",
    "bright": "\033[1m",
    "invert": "\033[7m",
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "default": "\033[39m"
}

log = None
log_lock = threading.Lock()


def get(fname=None):
  """
  Returns a logger instance, with optional log file output.
  """
  global log
  if log is not None and fname is None:
    return log

    # fname = os.environ.get("LOGTO", None)
    # if fname is None:
    #     fname = default_fname
  else:
    log = Logger(fname)

    return log


class Logger(object):

  def __init__(self, filename=None, default_verbose=0):
    """
    Constructs a logger with optional log file output.

    Args:
        filename: optional log file output. If None, nothing will be 
        written to file
    """
    now = datetime.datetime.now()
    self.verbose_thresh = int(os.environ.get("VERBOSE", 0))
    self.default_verbose = default_verbose
    if filename is not None:
      self.filename = filename
      dirname = os.path.dirname(self.filename)
      if not os.path.exists(dirname):
        os.makedirs(dirname)
      open(self.filename, "w").close()
      self.info("Log written to {}".format(os.path.abspath(self.filename)))
    else:
      self.filename = None

    pass

  @staticmethod
  def get_time_str(t=None):
    """
    Returns a formatted time string.

    Args:
        t: datetime, default now.
    """
    if t is None:
      t = datetime.datetime.now()
    timestr = t.isoformat(chr(32))
    return timestr

  def log(self, message, typ="info", verbose=None):
    """
    Writes a message.

    Args:
        message: string, message content.
        typ: string, type of the message. info, warning, error, or fatal.
        verbose: number, verbose level of the message. If lower than the 
        environment variable, then the message will be logged to standard 
        output and log output file (if set).
    """
    threadstr = "{}".format(threading.current_thread().ident)[-4:]
    if typ == "info":
      typstr_print = "{}I{}{}".format(TERM_COLOR["green"], threadstr,
                                      TERM_COLOR["default"])
      typstr_log = "I{}".format(threadstr)
    elif typ == "warning":
      typstr_print = "{}W{}{}".format(TERM_COLOR["yellow"], threadstr,
                                      TERM_COLOR["default"])
      typstr_log = "W{}".format(threadstr)
    elif typ == "debug":
      typstr_print = "{}D{}{}".format(TERM_COLOR["yellow"], threadstr,
                                      TERM_COLOR["default"])
      typstr_log = "D{}".format(threadstr)
    elif typ == "error":
      typstr_print = "{}E{}{}".format(TERM_COLOR["red"], threadstr,
                                      TERM_COLOR["default"])
      typstr_log = "E{}".format(threadstr)
    elif typ == "fatal":
      typstr_print = "{}F{}{}".format(TERM_COLOR["red"], threadstr,
                                      TERM_COLOR["default"])
      typstr_log = "F{}".format(threadstr)
    else:
      raise Exception("Unknown log type: {0}".format(typ))
    timestr = self.get_time_str()
    for (frame, filename, line_number, function_name, lines, index) in \
            inspect.getouterframes(inspect.currentframe()):
      fn = os.path.basename(filename)
      if fn != "logger.py":
        break
    cwd = os.getcwd()
    if filename.startswith(cwd):
      filename = filename[len(cwd):]
    filename = filename.lstrip("/")

    callerstr = "{}:{}".format(filename, line_number)
    if len(callerstr) > 20:
      callerstr = "...{}".format(callerstr[-17:])
    printstr = "{} {} {} {}".format(typstr_print, timestr, callerstr, message)
    logstr = "{} {} {} {}".format(typstr_log, timestr, callerstr, message)

    print(printstr)
    pass

  def log_wrapper(self, message, typ="info", verbose=None):
    if verbose is None:
      verbose = self.default_verbose

    if type(verbose) != int:
      raise Exception("Unknown verbose value: {}".format(verbose))

    log_lock.acquire()
    try:
      if self.verbose_thresh >= verbose:
        self.log(message, typ=typ, verbose=verbose)

      if self.filename is not None:
        with open(self.filename, "a") as f:
          f.write(logstr)
          f.write("\n")
    except e:
      print("Error occurred!!")
      print(str(e))
    finally:
      log_lock.release()

  def info(self, message, verbose=None):
    """
    Writes an info message.

    Args:
        message: string, message content.
        verbose: number, verbose level.
    """
    self.log_wrapper(message, typ="info", verbose=verbose)
    pass

  def warning(self, message, verbose=1):
    """
    Writes a warning message.

    Args:
        message: string, message content.
        verbose: number, verbose level.
    """
    self.log_wrapper(message, typ="warning", verbose=verbose)
    pass

  def error(self, message, verbose=0):
    """
    Writes an info message.

    Args:
        message: string, message content.
        verbose: number, verbose level.
    """
    self.log_wrapper(message, typ="error", verbose=verbose)
    pass

  def debug(self, message, verbose=None):
    self.log_wrapper(message, typ="debug", verbose=verbose)
    pass

  def fatal(self, message, verbose=0):
    """
    Writes a fatal message, and exits the program.

    Args:
        message: string, message content.
        verbose: number, verbose level.
    """
    self.log_wrapper(message, typ="fatal", verbose=verbose)
    sys.exit(0)
    pass

  def log_args(self, verbose=None):
    self.info("Command: {}".format(" ".join(sys.argv)))
    pass

  def log_exception(self, exception):
    tb_str = traceback.format_exc(exception)
    self.error(tb_str)
    pass

  def verbose_level(self, level):

    class VerboseScope():

      def __init__(self, logger, new_level):
        self._new_level = new_level
        self._logger = logger
        pass

      def __enter__(self):
        self._restore = self._logger.default_verbose
        self._logger.default_verbose = self._new_level
        pass

      def __exit__(self, type, value, traceback):
        self._logger.default_verbose = self._restore
        pass

    return VerboseScope(self, level)
