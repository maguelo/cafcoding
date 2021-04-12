import glob
import pandas as pd
from cafcoding.constants import headers_meteo

def load_meteo_dataframe(path, colnames=headers_meteo):
    all_files = glob.glob(path + "/*.csv")
    df_list = [pd.read_csv(filename, index_col=None, delimiter=';', header=None) for filename in all_files]
    df = pd.concat(df_list, axis=0, ignore_index=True)
    df.columns = colnames
    return df


def create_stations_list(df):
    stations_list =[]
    for station in df.estacion.unique():
        station_coord_obj = {'id': station, 
                            'long': df[df['estacion'] == station]['longitud'].max(), 
                            'lat':df[df['estacion'] == station]['latitud'].max()}
        stations_list.append(station_coord_obj)
    return stations_list
