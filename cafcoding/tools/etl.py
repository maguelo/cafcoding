import cafcoding.constants as constants

import numpy as np
import pandas as pd
import sys
import math
import geopy.distance 
from geographiclib.geodesic import Geodesic
import logging 
logger = logging.getLogger(constants.LOGGER_ID)

from cafcoding.constants import workspace_coordenates as ws_coord
from cafcoding.constants import PLC_MCP_CATEGORICAL

def filter_dataframe_by_date(df, field,min_date, max_date):
    return df[df[field].isin(pd.date_range(min_date, max_date))]

def calculate_distance(latitud, longitud, latitud_1, longitud_1):
    distance = geopy.distance.geodesic((latitud, longitud), (latitud_1, longitud_1))
    return distance.meters

def calc_nearest_station(list_stations,longitud, latitud):
    nearestDistance = sys.float_info.max  #Valor por defecto
    nearestStation = ''
    for station in list_stations:
        currentDistance = geopy.distance.geodesic((latitud, longitud), (station['lat'], station['long']))
        if currentDistance < nearestDistance:
            nearestDistance = currentDistance
            nearestStation = station
    return nearestStation

def calc_train_direction(lat1, long1, lat2, long2):
    """
    Esta funcion devuelve la direccion en grados
    """
    geod = Geodesic.WGS84
    g = geod.Inverse(lat1, long1, lat2, long2)
    return g['azi1']


def front_aerodinamic_wind(module, wing_dir, train_dir):
    """
    Fuerza aerodinamica del viento en el eje de la direccion
    :param module: Modulo del vector = velocidad del viento 
    """ 
    return module * math.cos(wing_dir - train_dir)

def lateral_aerodinamic_wind(module, wing_dir, train_dir):
    return module * math.sin(wing_dir - train_dir)

def calculate_percent_slope(altitude, prev_altitude, distance):
    return ((altitude - prev_altitude) / distance) * 100 if distance != 0 else np.nan


#Funciones relacionadas con los datos de CAF

def column_to_absolute(df,columns, drops_original = True):
    """
    Esta funcià¸£à¸“n crea columna de valores absolutos y da la opcion de borrar la original
    """
    if columns is None:
        columns = []
        
    for column in columns:
        df[column+'_abs'] = df[column].apply(lambda x: abs(x))
        
    if drops_original:
        df = df.drop(columns,axis = 1)
    return df

def create_shifts(df,columns_to_shift):
    """
    Con esta funcion se crean las columnas con el valor anterior para mejorar el modelo
    """
    if columns_to_shift is None:
        columns_to_shift = []
    for col in columns_to_shift:
        df[col+'_1']=df[col].shift(periods=+1)
    return df

def create_differences(df,columns_to_shift):
    """
    se crean las diferencias enntre la columna y la columna -1
    """
    if columns_to_shift is None:
        columns_to_shift = []
    for col in columns_to_shift:
        if col+'_prev' not in df.columns:
            df = create_shifts(df,columns_to_shift)

        df[col+'_dif']=df[col] - df[col+'_1']
    return df

#----------------------------------------------------------------------------------


def adjust_freq_in_df(df, delta_freq):
    return df.asfreq(delta_freq)

def fill_dataframe_by_ut(df, delta_freq=None):
    """
    Rellenamos dataframe propagando primero hacia atras y luego hacia delante
    """
    # Evita el error de chained _assignment, la copia la estamos haciendo sobre si mismos por lo que no deberia de haber problema
    pd.set_option('mode.chained_assignment', None)
    logger.debug("In: %s ",", ".join([str(date) for date in df.date_day.unique()]))    
    
    list_df = []
    for day in df.date_day.unique():
        for ut in df.ut.unique():
            df_tmp = df.loc[(df.date_day==day) & (df.ut==ut),]
            if delta_freq is not None:
                df_tmp= adjust_freq_in_df(df_tmp,delta_freq)
            df_tmp.fillna(method='bfill', inplace=True)
            df_tmp.fillna(method='ffill', inplace=True)  
            list_df.append(df_tmp)
    logger.debug("Out: %s ",", ".join([str(date) for date in df.date_day.unique()]))    
    
    # Restauramos el valor original para que nos avise
    pd.set_option('mode.chained_assignment', 'warn')
    return pd.concat(list_df)



def check_longitude(value):
    if value < ws_coord['LON_MIN']:
        return np.nan
    if value > ws_coord['LON_MAX']:
        return np.nan
    return value 

def check_latitude(value):
    if value < ws_coord['LAT_MIN']:
        return np.nan
    if value > ws_coord['LAT_MAX']:
        return np.nan
    return value
    
    
    
# Conversiones categoricas

def convertStr2float(data, columns):
    if columns is None:
        columns = []
        
    for col in columns:
        data[col] = data[col].apply(lambda x: float(x.replace(',','.')))
    return data
  
def convert_boolean(data, columns):
    if columns is None:
        columns = []
        
    for col in columns:
        data[col] = data[col].apply(lambda x: 1 if x else 0)
    return data
    
def plc_master_controller_pos_2_categorical(data, plc_mcp_dict=PLC_MCP_CATEGORICAL):
    for key in plc_mcp_dict:
        data[plc_mcp_dict[key]] = np.where(data['PLC_MASTER_CONTROLLER_POS']==key,1,0)
    return data
    
def calculate_slope_as_categorical(altitude, prev_altitude):
    return (altitude - prev_altitude) and (1, -1)[(altitude - prev_altitude) < 0]