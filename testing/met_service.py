# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:37:08 2020

@author: michaelek
"""

import os
import xarray as xr
import pandas as pd

pd.options.display.max_columns = 10


#######################################
### Parameters

base_path = r'N:\met_service\forecasts'

nc1 = 'wrf_hourly_precip_nz4kmN-NCEP_2020042712.nc'

#######################################
### Import data

ds1 = xr.open_dataset(os.path.join(base_path, nc1))







