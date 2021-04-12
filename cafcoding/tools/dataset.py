from cafcoding.stages import etl_step
import cafcoding.tools.log as log
from cafcoding import constants

import logging 
logger = logging.getLogger(constants.LOGGER_ID)

import pandas as pd

class DatasetLoader(object):
    PATH_TEMPLATE_DRIVE='/content/drive/MyDrive/CAFcoding/dataset/{step}/2020-{month:02d}-{day:02d}-etl-{version}.csv'
    PATH_TEMPLATE_S3='s3://cafcodingdatos/targets_mirror/{step}/2020-{month:02d}-{day:02d}-etl-{version}.csv'
    
    def __init__(self, path_template='s3', etl_version=None):
        
        if path_template == 's3' or path_template is None:
            self.target = 's3'
            self.path_template = self.PATH_TEMPLATE_S3
            
        elif path_template == 'drive':
            self.target = 'drive'
            self.path_template = self.PATH_TEMPLATE_DRIVE
            
        else:
            self.target = 'path'
            self.path_template = path_template
            
        logger.info("Using datasource from {} {}".format(self.target, self.path_template))
            
        # self.path_template = self.PATH_TEMPLATE if path_template is None else path_template
        self.etl_version = etl_step.ETL_VERSION if etl_version is None else etl_version
        self.dataset_conf = {"train":{"start_date":1, "days":31, "month":10 },
                             "val":{"start_date":1, "days":10 , "month":11 },
                             "test":{"start_date":11, "days":30 , "month":11 }}
        
        self.dataset_files={key:[] for key in self.dataset_conf}
        self.prepare_dataset_files()

    def prepare_dataset_files(self):
        for key in self.dataset_conf:
            for day in range(self.dataset_conf[key]["start_date"],self.dataset_conf[key]["days"]+1):
                self.dataset_files[key].append(self.path_template.format(step=key, 
                                                                         month=self.dataset_conf[key]['month'], 
                                                                         day=day, 
                                                                         version=self.etl_version))
                
    def load_df_train(self, number_days= None):
        return self._load_df_dict("train",number_days)
        
    def load_df_val(self, number_days= None):
        return self._load_df_dict("val",number_days)
    
    def load_df_test(self, number_days= None):
        return self._load_df_dict("test",number_days)

    def _load_df_dict(self, step, number_days):
        number_days = self.dataset_conf[step]["days"] if number_days is None else min(number_days, self.dataset_conf[step]["days"])
        df=[]
        for path in self.dataset_files[step][:number_days]:
            try:
                print (path)
                df_temp = pd.read_csv(path,index_col=0)
                df.append(df_temp)
            except Exception as ex:
                logger.error("{0}: {1}", path, str(ex))
        return pd.concat(df)
