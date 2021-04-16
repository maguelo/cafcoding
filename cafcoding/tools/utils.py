import os
import time
import datetime
from cafcoding.tools import log

import logging 

logger = logging.getLogger('ETL')

def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        logger.error("Creation of the directory %s failed" % path)
    else:
        logger.error("Successfully created the directory %s " % path)
        
def get_timestamp():
    today = datetime.datetime.fromtimestamp(time.time())
    return today.strftime("%Y%m%d_%H%M%S")        

