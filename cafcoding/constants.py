# coordenadas de trabajo 
import numpy as np

LOGGER_ID = "ETL"

LOG_FILE= 'cafcoding'
LOG_DIR = '.'

workspace_coordenates = {'LAT_MIN': 41, 
                         'LAT_MAX': 44,
                         'LON_MIN': 0,
                         'LON_MAX': 4}


# PLC_MASTER_CONTROLLER_POS
PLC_MCP_CATEGORICAL={0:"plc_mcp_urgency_brake_requested",
                    1:"plc_mcp_braking_effort_requested",
                    2:"plc_mcp_coasting_requested",
                    3:"plc_mcp_propulsion_effort_requested",
                    4:"plc_mcp_error_unknown_requested"}


headers_meteo=['estacion', 'longitud', 'latitud', 'id', 'ciudad', 'fecha', 'precipitacion_pluviometro_acu', 'precipitacion_disdrometro_acu', 'precipitacion_liquida_acu', 'precipitacion_solida_acu', 'wind_vmax_3s', 'wind_vmean_10m', 'wind_max_ultrasound_10m', 'wind_direction_ultrasound_10m', 'wind_direction_10m', 
          'wind_direction_ultrasound_10m', 'wind_max_direction_60m', 'wind_ultrasound_direction_60m', 'svd_wind_10m', 'svd_wind_direction_10m', 'svd_wind_ultrasonic_10m', 'svd_wind_ultrasonic_direction_10m', 'hri', 'insolation_duration_1h', 'presion_barometro', 'presion_reducida_750', 'temp_ground_10m', 'temp_subground20_10m', 
          'temp_subground5_10m', 'tia', 'temp_rocio', 'tmin', 'tmax', 'visibility10', 'ref_alt_baro_750', 'ref_alt_baro_850', 'ref_alt_baro_925', 'wind_route_60', 'snow_60'] 

    
