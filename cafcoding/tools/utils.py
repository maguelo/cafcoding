import os
from cafcoding.tools import log

import itertools
import time
import datetime
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

def iterator_product(*args):
    """
    Generate all possibilities to mix N list of params.
    Return list with all combos and len of if
    """
    product_len=1
    for param in args:
        product_len*=len(param)
    return itertools.product(*args), product_len