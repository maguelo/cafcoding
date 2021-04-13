# #https://medium.com/@lily_su/accessing-s3-bucket-from-google-colab-16f7ee6c5b51
import os
import s3fs

class AWS(object):
    def __init__(self, path_config):
        self.path_config= path_config
        self.fs = s3fs.S3FileSystem()
        
    def generate_session_file(self,service_name,key_id,secret_key,region_name):
        text = f'''
                [default]
                service_name= s3
                aws_access_key_id = {key_id} 
                aws_secret_access_key = {secret_key}
                region_name= {region_name}
                ''' 
        with open(self.path_config, 'w') as f:
            f.write(text)
    
    def set_credentials_in_env(self):
        os.environ['AWS_SHARED_CREDENTIALS_FILE'] = self.path_config
        print(os.environ['AWS_SHARED_CREDENTIALS_FILE'])
        
    def ls(self, path):        
        return self.fs.ls(path)      
