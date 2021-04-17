from cafcoding import constants


import functools
import logging
import os, sys
import colorlog

def get_logger(log_file_name=constants.LOG_FILE,log_dir=constants.LOG_DIR):
    """ Creates a Log File and returns Logger object """
    
    # Build Log File Full Path
    logPath = log_file_name if os.path.exists(log_file_name) else os.path.join(log_dir, (str(log_file_name) + '.log'))

    # Create logger object and set the format for logging and other attributes
    #logger = logging.Logger(log_file_name)
    logger = logging.getLogger(constants.LOGGER_ID)
    logger.setLevel(logging.DEBUG)
    

    fh = logging.FileHandler(logPath)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
    fh.setFormatter(formatter)
    sh.setFormatter(colorlog.ColoredFormatter('%(log_color)s [%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S'))
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Return logger object
    return logger
    
#logger = get_logger()

def log_decorator(func):
    
    def log_decorator_wrapper(self, *args, **kwargs):
        # Build logger object
        logger = logging.getLogger(constants.LOGGER_ID)
        logger.debug(f"{func} START")
        
        value = func(self, *args, **kwargs)
        logger.debug(f"{func} END")
        return value
    # Return the pointer to the function
    return log_decorator_wrapper

