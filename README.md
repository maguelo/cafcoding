# cafcoding

## To Install

pip install git+https://github.com/maguelo/cafcoding.git#egg=cafcoding


## Generate aws creedentials

import cafcoding.tools.aws as aws

amazon = aws.AWS("PATH DE VUESTRO EQUIPO/awscli.ini")

amazon.generate_session_file('s3', key_id, secret_key ,'eu-west-1')

!cat "PATH DE VUESTRO EQUIPO/awscli.ini"

### Activamos las credenciales

amazon.set_credentials_in_env()
