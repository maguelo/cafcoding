from cafcoding.tools import etl
from cafcoding.tools import meteo
from cafcoding.tools import log
from cafcoding import constants

from pandarallel import pandarallel

import pandas as pd
import srtm
import numpy as np
import logging 
logger = logging.getLogger(constants.LOGGER_ID)

pandarallel.initialize()


ETL_VERSION = "1.2.2"

ABS_COLUMNS = ['TCU1_Axle1Speed','TCU1_Axle2Speed','TCU1_ElecEffApp',
    'TCU2_Axle1Speed','TCU2_Axle2Speed','TCU2_ElecEffApp',
    'TCU3_Axle1Speed','TCU3_Axle2Speed','TCU3_ElecEffApp',
    'TCU4_Axle1Speed','TCU4_Axle2Speed','TCU4_ElecEffApp']

SHIFT_COLUMNS = ['TCU1_LinePowerConsumed','TCU1_LinePowerDissipated', 'TCU1_LinePowerReturned',
    'TCU2_LinePowerConsumed','TCU2_LinePowerDissipated', 'TCU2_LinePowerReturned',
    'TCU3_LinePowerConsumed','TCU3_LinePowerDissipated', 'TCU3_LinePowerReturned',
    'TCU4_LinePowerConsumed','TCU4_LinePowerDissipated', 'TCU4_LinePowerReturned',
    'TCU1_DCBusVoltage','TCU2_DCBusVoltage','TCU3_DCBusVoltage','TCU4_DCBusVoltage',
    'PLC_TRACTION_BRAKE_COMMAND','PLC_Speed','EPAC1_WSP_Acting','EPAC2_WSP_Acting',
    'TCU1_Axle1Speed_abs','TCU1_Axle2Speed_abs','TCU1_ElecEffApp_abs',
    'TCU2_Axle1Speed_abs','TCU2_Axle2Speed_abs','TCU2_ElecEffApp_abs',
    'TCU3_Axle1Speed_abs','TCU3_Axle2Speed_abs','TCU3_ElecEffApp_abs',
    'TCU4_Axle1Speed_abs','TCU4_Axle2Speed_abs','TCU4_ElecEffApp_abs']
    
TRAIN_REMOVE_AT_STOP_WINDOW = 5

@log.log_decorator
def process_etl(df, df_meteo, del_stopped_train=True, columns_to_abs=ABS_COLUMNS, columns_to_shift=SHIFT_COLUMNS, fill_delta= False):
    
    df= prepare_data_time(df)
    date_range = {'min': df['ts_date'].min(), 'max': df['ts_date'].max()}
    
    logger.info('Filter dates: {} - {}'.format(date_range["min"],date_range["max"]))
    
    df_meteo, stations_list = process_meteo_dataframe(df_meteo,date_range)
    
    

    df = etl.fill_dataframe_by_ut(df)

    df = generate_new_columns_from_gps_data(df, stations_list)

    df = merge_with_meteo(df, df_meteo)

    df, delta = create_index_ts_date(df)

    if not fill_delta:
        delta = None

    # Cuidado realizamos un fill_dataframe_by_ut aplicando delta
    df = generate_auxiliar_columns(df,delta)

    if del_stopped_train:
        df = delete_stopped_train(df)
    
    # Funciones de contexto fisico
    df = etl.column_to_absolute(df,columns_to_abs)
    df = etl.create_shifts(df,columns_to_shift)
    df = etl.create_differences(df,columns_to_shift)

    df= create_categorical(df)


    df = etl.fill_dataframe_by_ut(df)
    
    return df

@log.log_decorator
def prepare_data_time(df):
    df = df.sort_values(by=['ut','ts_date'])
    df['ts_date'] = pd.to_datetime(df.ts_date,format="%Y/%m/%d %H:%M:%S.%f")
    df['date_day'] =df.ts_date.dt.date
    return df

@log.log_decorator
def process_meteo_dataframe(df_meteo, date_range, columns_2_drop=['id', 'ciudad', 'temp_rocio', 'longitud', 'latitud'],null_threshold = 0.4):
    
    #Primero eliminamos los rows que no nos interesan de df_meteo
    df_meteo['fecha'] = pd.to_datetime(df_meteo.fecha,format="%Y/%m/%d %H:%M:%S.%f")
    df_meteo = etl.filter_dataframe_by_date(df_meteo, 'fecha', date_range['min'], date_range['max'])

    #Nos quedamos solo con los que tengamos suficientes datos
    df_meteo = df_meteo.loc[:, df_meteo.notnull().mean() > null_threshold]
    #df_meteo = df_meteo.loc[:, df_meteo.isnull().mean() < null_threshold]

    stations_list=meteo.create_stations_list(df_meteo)

    columns_2_drop = list(set(columns_2_drop).intersection(df_meteo.columns))
    df_meteo = df_meteo.drop(columns_2_drop, axis=1)
    
    return df_meteo, stations_list


COLUMNS_2_DROP_GPS_DATA_STEP = ['SI_GPS_LonG', 'SI_GPS_LonM', 'SI_GPS_LonS', 'SI_GPS_LatG', 'SI_GPS_LatM', 'SI_GPS_LatS','SI_GPS_LatFracS',  'SI_GPS_LonFracS']


@log.log_decorator
def generate_new_columns_from_gps_data(df, stations_list, columns_to_drop=COLUMNS_2_DROP_GPS_DATA_STEP):
    
    #Presuponemos que los datos fuera de la zona de interés son erróneos y serán más adelante rellenados
    df['SI_GPS_LonG'] = df.apply(lambda x:etl.check_longitude(x['SI_GPS_LonG']), axis=1)
    df['SI_GPS_LatG'] = df.apply(lambda x:etl.check_latitude(x['SI_GPS_LatG']), axis=1)

    df['Longitud'] = -sum([df.SI_GPS_LonG, (df.SI_GPS_LonM/60), (df.SI_GPS_LonS/3600)]) #Inversa porque está medido de E -> O
    df['Latitud'] = sum([df.SI_GPS_LatG, (df.SI_GPS_LatM/60), (df.SI_GPS_LatS/3600)])

    df = df.sort_values(by=['ut','ts_date'])
    df = etl.fill_dataframe_by_ut(df)

    df['Nearest_Station'] = df.apply(lambda r : etl.calc_nearest_station(stations_list,r.Longitud, r.Latitud)['id'], axis=1)

    
    df.drop(columns_to_drop, axis=1, inplace=True)

    return df

@log.log_decorator
def merge_with_meteo(df, df_meteo):
    
    df_meteo['Nearest_Station'] = df_meteo['estacion']
    df_meteo['ts_date'] = df_meteo['fecha']


    df_meteo = df_meteo.sort_values(by=['Nearest_Station','ts_date'])
    df_meteo = df_meteo.set_index(['Nearest_Station','ts_date'])

    df = df.join(df_meteo, on=["Nearest_Station", "ts_date"])

    return df

@log.log_decorator
def create_index_ts_date(df):
    # Guardamos ts_date para poder determinar la frequencia mas tarde
    list_ut = list(df.ut.unique())
    ts_date = df[df.ut==list_ut[0]]['ts_date']
    df.set_index('ts_date', inplace=True)
    df['ts_date'] = df.index
    delta = ts_date.diff().min()
    
    return df, delta

@log.log_decorator
def generate_auxiliar_columns(df,delta):
    df['Latitud-1']=df["Latitud"].shift(periods=+1) # Sentido 1 o -1
    df['Longitud-1']=df["Longitud"].shift(periods=+1)

    df["Direction"] = df.apply(lambda r: etl.calc_train_direction(r["Latitud-1"],
                                                                r["Longitud-1"],
                                                                r["Latitud"],
                                                                r["Longitud"]), axis=1)

    df['fr_wind'] = df.apply(lambda x: etl.front_aerodinamic_wind(x['wind_vmean_10m'], x['wind_direction_10m'], x['Direction']), axis=1)
    df['lat_wind'] = df.apply(lambda x: etl.lateral_aerodinamic_wind(x['wind_vmean_10m'], x['wind_direction_10m'], x['Direction']), axis=1)

    #Rellenamos los valores tras el merge
    df = etl.fill_dataframe_by_ut(df,delta)

    #Alturas
    elevation_data = srtm.get_data()

    df['altitude'] = df.apply(lambda x: elevation_data.get_elevation(x['Latitud'], x['Longitud']), axis=1)
    df['altitude-1'] = df.apply(lambda x: elevation_data.get_elevation(x['Latitud-1'], x['Longitud-1']), axis=1)
    df['slope'] = df.apply(lambda x: etl.calculate_slope_as_categorical(x['altitude'], x['altitude-1']), axis=1)
    df['distance'] = df.apply(lambda x: etl.calculate_distance(x['Latitud'], x['Longitud'], x['Latitud-1'], x['Longitud-1']), axis=1)
    df['percent_slope'] = df.apply(lambda x: etl.calculate_percent_slope(x['altitude'], x['altitude-1'], x['distance']), axis=1)

    columns_to_drop = ['Norte', 'Este', 'Nearest_Station',
                   'Direction', 'estacion', 'fecha', 'wind_vmax_3s', 'wind_vmean_10m', 'wind_direction_10m',
                   'wind_max_direction_60m', 'tia', 'tmin', 'tmax', 'Latitud-1', 'Longitud-1',  'altitude-1'] #'Latitud', 'Longitud'

    df.drop(columns_to_drop, axis=1, inplace=True)
    
    return df



@log.log_decorator
def delete_stopped_train(df , windows_size=10, col_target='PLC_Speed'):
    def is_train_stopped(row, target_value):
        return (row == target_value).all()

    windows_target = [col_target+'_'+str(period) for period in range(windows_size+1)]
    target_values = [ 0.0 for col in windows_target]

    list_ut_df = []
    for ut in df['ut'].unique():
        df_ut = df[df['ut']==ut].copy()
        
        for period,col in enumerate(windows_target):
            df_ut[col]=df_ut[col_target].shift(periods=period)

        # Eliminamos datos sucios, a mayor tamano de ventana mayor perdida de datos
        df_ut=df_ut[windows_size:]
        
        df_ut['train_stopped']= df_ut.parallel_apply(lambda row:is_train_stopped(row[windows_target], target_values),axis=1)
        df_ut= df_ut[df_ut['train_stopped']==False]
        

        list_ut_df.append(df_ut)

    df= pd.concat(list_ut_df)
    df.drop(windows_target+['train_stopped'],axis=1,inplace=True)
    
    return df

@log.log_decorator
def create_categorical(df, ignore_columns= ['ut','ts_date']):
    
    object_2_float = list(df.columns)
    for col in ignore_columns:
        object_2_float.remove(col)

    dict_types = {}
    for field in object_2_float:
        if not df[field].dtypes in dict_types:
            dict_types[df[field].dtypes]=[]
        dict_types[df[field].dtypes].append(field)

    df = etl.convert_boolean(df,dict_types[np.dtype('bool')])
    df = etl.plc_master_controller_pos_2_categorical(df)
    
    return df
