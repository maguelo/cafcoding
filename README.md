# cafcoding

## PPT Resumen práctica
/PPT Entrega práctica

## Data Visualization with D3.js
https://codepen.io/danielvillalba/full/oNBqLpM

## Trello Scrum Metodology
https://trello.com/b/PCLD5j59/scrum-board

## To Install the Library

pip install git+https://github.com/maguelo/cafcoding.git#egg=cafcoding


## How to Generate aws creedentials

import cafcoding.tools.aws as aws

amazon = aws.AWS("PATH DE VUESTRO EQUIPO/awscli.ini")

amazon.generate_session_file('s3', key_id, secret_key ,'eu-west-1')

!cat "PATH DE VUESTRO EQUIPO/awscli.ini"

### Creedentials Activation

amazon.set_credentials_in_env()
