#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:34:07 2024

@author: pfountou
"""

''' ------ Import modules ------ '''

import os
import sys
import glob
import datetime as dtt # from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc # from netCDF4 import Dataset
from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib import animation
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm
import matplotlib.colors
from matplotlib.cm import get_cmap
from matplotlib import cm as plt_cm
from cmcrameri import cm
import h5py
import sentinelsat ### A warning pops up!!!!!
import harp
import coda
import functools
import math
import time
import pytz
from scipy.stats import gaussian_kde
from scipy.interpolate import Akima1DInterpolator
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = UserWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
import io
import unicodedata
import string
import random
from collections import deque
import pickle # ...
import joblib # Save and load models
import cdsapi
import wget
import requests
import hashlib
from pystac import Collection
from pystac_client import ItemSearch

# import airbase # To download data from EEA - Need installation!!
# import xgboost as xgb # The XGBoost model - Need installation!!
# import cv2 # For image processing (not important) - Need installation!!
# import graphviz # For visualizing Decision Trees (not important) - Need installation!!


import torch as tch
import torch.nn as tnn # Neural Network
import torch.nn.functional as funtnn
from torch.utils.data import Dataset, DataLoader
import torch.optim as tchopt
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torchvision as tchvis
import torchvision.transforms as tvistransforms
from torchvision import datasets, models, transforms
import torchmetrics.regression, torchmetrics.classification
from torch import multiprocessing as mp
# import pytorch_lightning as ptl # Need installation!!
# from pytorch_lightning import Trainer


import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D,\
    MaxPooling2D, LSTM, BatchNormalization, Input # , CuDNNLSTM
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy,\
    BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# from tensorflow.keras.datasets import CIFAR10 # or any other available dataset (Problem!!!)


from sklearn import datasets, svm, preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder,\
    OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold,\
    StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit, RepeatedKFold,\
        RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, f1_score,\
    precision_score, recall_score, mean_squared_error, mean_absolute_error,\
        ConfusionMatrixDisplay
from sklearn import tree # Export a decision tree in a graphical format
from sklearn.datasets import make_blobs, make_regression, load_iris # Iris flower dataset
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,\
    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA


from skorch import NeuralNet, NeuralNetRegressor, NeuralNetBinaryClassifier,\
    NeuralNetClassifier
from skorch.callbacks import EpochScoring


''' ------ API related functions ------ '''

def download_AERONET(url_par, st_nm, start_date, end_date, date_fmt, product_pars,
                     inv_par, out_fl):
    
    '''
    Download a file from AERONET (example from Dust Training course, 2023).

    INPUTS:
    url_par: the main url of the AERONET site (str)
    st_nm: the code name of the station (str)
    start_date, end_date: the start and end raw date (str, e.g. '20180320')
    date_fmt: the format of the dates (str, e.g. '%Y%m%d', '%Y%m%d%H%M%S, etc.')
    
    product_pars: list which contains info about the product that we want to
    download.
    1) If we want the AOD/SDA products, product_pars = [atm_par, avg_par] where:
    atm_par --> the atmospheric product we want to download (str, e.g. 'AOD10',
    'AOD20', 'SDA10', 'TOT15', etc.) and
    avg_par --> either all or daily points of the measurements (int, 10 or 20
    respectively)
    2) If we want the inversion products (e.g. SIZ, SSA, etc.),
    product_pars = [pr_id, avg_par, inv_type] where:
    pr_id --> the ID of the product we want to download (str, e.g. 'SIZ', 'SSA', etc.)
    avg_par --> either all or daily points of the measurements (int, 10 or 20
    respectively)
    inv_type --> the type of the inversion (str, e.g. 'ALM20', 'HYB15', etc.)
    
    inv_par: choose if you want the AOD/SDA data, or the inversion ones, e.g.
    SIZ, SSA, etc. ('yes', or 'no')
    out_fl: the output folder (or file name) where the file will be saved (str)
    
    OUTPUTS:
    the AERONET file
    
    NOTES:
    In this function, we can also add "hour" in the raw dates. For the data type,
    other options are AOD10, AOD20, SDA10, SDA15 etc., in which the number indicates
    the level of the data. For example, the AOD10 stands for the AOD level 1.0 data.
    The "AVG" parameter defines the data format as: AVG=10 --> all points and
    AVG=20 --> daily averages.
    For more info about the AOD/SDA products see:
    'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3'.
    For more info about the inversion products (e.g. SIZ, SSA, etc.) see:
    'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3'.
    We can modify this function to add more products based on the examples
    in the following link:
    'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3'
    '''
    
    ## Convert the raw dates into datetime ##
    start_date_dt = dtt.datetime.strptime(start_date, date_fmt)
    end_date_dt = dtt.datetime.strptime(end_date, date_fmt)
    
    ## Define the dictionary with the file parameters/Make the URL ##
    
    if inv_par == 'no': # for AOD/SDA products (and for other products, as well)
        
        atm_par = product_pars[0]; avg_par = product_pars[1]
        
        data_dict = {
            'endpoint': url_par,
            'station': st_nm,
            'year': start_date_dt.year, 'month': start_date_dt.month,
            'day': start_date_dt.day,
            'year2': end_date_dt.year, 'month2': end_date_dt.month,
            'day2': end_date_dt.day,
            atm_par: 1, 'AVG': avg_par}
        
        url = '&'.join(['{endpoint}?site={station}',
                        'year={year}', 'month={month}', 'day={day}',
                        'year2={year2}', 'month2={month2}', 'day2={day2}',
                        atm_par+'={'+atm_par+'}',
                        'AVG={AVG}']).format(**data_dict)
    
    else: # only for inversion products
    
        pr_id = product_pars[0]; avg_par = product_pars[1]; inv_type = product_pars[2]
        
        data_dict = {
            'endpoint': url_par,
            'station': st_nm,
            'year': start_date_dt.year, 'month': start_date_dt.month,
            'day': start_date_dt.day,
            'year2': end_date_dt.year, 'month2': end_date_dt.month,
            'day2': end_date_dt.day,
            'product': pr_id, 'AVG': avg_par, inv_type: 1}
        
        url = '&'.join(['{endpoint}?site={station}',
                        'year={year}', 'month={month}', 'day={day}',
                        'year2={year2}', 'month2={month2}', 'day2={day2}',
                        'product={product}', 'AVG={AVG}',
                        inv_type+'={'+inv_type+'}']).format(**data_dict)
    
    ## Download the file ##
    wget.download(url, out_fl) # +'.txt'

def download_ERA5(key_par, data_dict, data_info_lst, area_lst,
                  st_nm, out_fl):
    
    '''
    This function can be used to download hourly meteorological CAMS ERA5
    Reanalysis data from the Copernicus CDS webpage (in NetCDF format). Each
    file can contain either one, or more meteorological parameters.
    
    INPUTS:
    key_par: the key for downloading (str)
    data_dict: dictionary which contains the names of the variable(s) we want
    to download and their reanalysis type (dict)
    data_info_lst: list of lists with info about the data (str, [[year], [month],
    [day], [time], year_range, [pressure]], [['2021', '2022'], ['01', '02'], ...]).
    Note that year_range is only one value, not a list!!
    area_lst: list of coordinates (int, e.g. [north edge, west edge,
    south edge, east edge] --> [71, -22, 43, 28])
    st_nm: the name of the station (str)
    out_fl: the output path/name of the file (str)
    
    OUTPUTS:
    the ERA5 file
    
    NOTES:
    To get the key_par and the url_par follow the steps from 1) to 3) from the
    function above.
    We can do any modifications to this function, to fit our needs.
    '''
    
    ## Get the values from the time_info_lst parameter ##
    yr_lst = data_info_lst[0]; month_lst = data_info_lst[1]
    day_lst = data_info_lst[2]; time_lst = data_info_lst[3]
    yrange = data_info_lst[4]; press_lst = data_info_lst[5]
    
    for var in data_dict.keys():
        
        retype = data_dict[var]
        # print(f'parameter {var} is downloading...')
        
        if '-single-' in retype:
            
            ## Make the main output path and the file's name ##
            outpath = out_fl+'Single_levels/'
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            
            file_nm = '_'.join([st_nm, var, yrange])
            file_out = outpath+file_nm
            
            ## Download the data ##
            c = cdsapi.Client(url = 'https://cds.climate.copernicus.eu/api/v2',
                              key = key_par)
            c.retrieve(
                retype, {'product_type': 'reanalysis', 'format': 'netcdf',
                         'variable': var, 'year': yr_lst, 'month': month_lst,
                         'day': day_lst, 'time': time_lst, 'area': area_lst},
                f'{file_out}.nc')
        
        else:
            
            ## Make the main output path ##
            outpath = out_fl+'Pressure_levels/'
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            
            for y in yr_lst:
                
                ## Make the year output path ##
                outpath_yr = outpath+f'{y}/'
                if not os.path.exists(outpath_yr):
                    os.mkdir(outpath_yr)
                
                for m in month_lst:
                    
                    ## Make the file's name ##
                    file_nm = '_'.join([st_nm, var, f'{y}{m}'])
                    file_out = outpath_yr+file_nm
                    
                    ## Download the data ##
                    c = cdsapi.Client(url = 'https://cds.climate.copernicus.eu/api/v2',
                                      key = key_par)
                    c.retrieve(
                        retype, {'product_type': 'reanalysis', 'format': 'netcdf',
                                 'variable': var, 'pressure_level': press_lst,
                                 'year': y, 'month': m, 'day': day_lst,
                                 'time': time_lst, 'area': area_lst}, f'{file_out}.nc')

def download_S5P_L2_PAL(product_id, date_list, area_list, pathout):
    
    '''
    This function downloads S5P L2 PAL file(s).
    
    INPUTS:
    product_id: the product name in the data collection (str)
    date_list: list/array with dates in format YYYY-mm-dd (str)
    area_list: list with the coordinates of the square's points for a specific
    area of interest (int/float, [north edge, west edge, south edge, east edge])
    pathout: the folder where the file(s) will be saved (str)
    
    OUTPUTS:
    the S5P L2 PAL file(s)
    
    NOTES:
    You can find the ID of the products here
    https://data-portal.s5p-pal.com/api/s5p-l2/collection.json
    
    Example that works, as well, but I do not understand the geofilter part:
    
    timefilter = "2021-01-01"
    geofilter = {
        'type': 'Polygon',
        'coordinates': [[[6.42425537109375, 53.174765470134616],
                          [7.344360351562499, 53.174765470134616],
                          [7.344360351562499, 53.67393435835391],
                          [6.42425537109375, 53.67393435835391],
                          [6.42425537109375, 53.174765470134616]]]}
    items = ItemSearch(endpoint, datetime = timefilter, intersects = geofilter).items()
    
    More info about the input parameters in the ItemSearch() function here
    https://pystac-client.readthedocs.io/en/latest/api.html#item-search
    '''
    
    url = f'https://data-portal.s5p-pal.com/api/s5p-l2/{product_id}/collection.json'
    collection = Collection.from_file(url)
    endpoint = collection.get_single_link('search').target
    
    for timefilter in date_list:
        
        items = ItemSearch(endpoint, datetime = timefilter, bbox = area_list).items()
        
        for item in list(items):
            
            download_url = item.assets['download'].href
            product_filename = item.properties['physical_name']
            product_hash = item.properties['hash']
        
            print(f'Downloading {product_filename}...')
            
            r = requests.get(download_url)
            with open(f'{pathout}{product_filename}', 'wb') as product_file:
                product_file.write(r.content)
            
            file_hash = 'md5:'+hashlib.md5(open(pathout+product_filename, 'rb').\
                                           read()).hexdigest()
            print('Checking hash...')
            
            assert file_hash == product_hash
            print('Product was downloaded correctly!!')


''' ------ Time related functions ------ '''

def utc_sec_to_datetime(sec_arr, timestamp_val):
    
    '''
    This function converts the UTC time (in seconds) into full
    date in datetime format.
    
    INPUTS:
    sec_arr: array of time values in seconds
    timestamp_val: the seconds between 01/01/1970 (Python start date)
                and the start date of the respective data
    
    OUTPUTS:
    dt_arr: array of time values in datetime format
    
    NOTES:
    The timestamp_val is the seconds between 01/01/1970 (Python start date)
    and the start date of the respective data. The formula is the following:
    1 yr in seconds*(data initial yr - python initial yr) +/- some days in seconds.
    For example, for S5P/TROPOMI the value is:
    timestamp_val = 31536000*(2010 - 1970) + 864000
    '''
    
    dt_arr = []
    for val in sec_arr:
        # temp_val = dtt.datetime.utcfromtimestamp(val+timestamp_val)
        dt_arr.append(dtt.datetime.utcfromtimestamp(val+timestamp_val))
    
    dt_arr = np.array(dt_arr)
    
    return dt_arr

def utc_to_local_tz(utc_dt_arr, local_tz):
    
    '''
    This function converts an array/list of datetime values (in UTC) to any local time zone. The local
    timezones can be found in https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568
    
    INPUTS:
    utc_dt_arr: array of time values in datetime format
    local_tz: any local time zone (str)
    
    OUTPUTS:
    local_dtdate: array of time values in the selected time zone in datetime format
    '''
    
    lcl_tz = pytz.timezone(local_tz)
    local_dtdate = [pytz.utc.localize(x).astimezone(lcl_tz) for x in utc_dt_arr]
    
    local_dtdate = np.array(local_dtdate) # Turn the list into an array
    
    return local_dtdate

def date_to_datetime(date_arr, date_format):
    
    '''
    This function converts raw dates into datetime format.
    
    INPUTS:
    date_arr: array of date values (int/str)
    date_format: the format of the date values (str, e.g. '%Y%m%d%H%M',
                or %Y/%m/%d %H:%M, etc.)
    
    OUTPUTS:
    dt_date_arr: array of dates in datetime format
    
    NOTES:
    Infos about datetime format can be found in:
    https://www.w3schools.com/python/python_datetime.asp
    '''
    
    dt_date_arr = []
    for date_value in date_arr:
        
        # Check if the value of the date_arr parameter are str.
        # If not, convert them into a str
        # if isinstance(date_value, str) != True:
        #     date_value = str(date_value)
    
        # Make the datetime
        dt_date_arr.append(dtt.datetime.strptime(str(date_value), date_format))
    
    dt_date_arr = np.array(dt_date_arr) # Turn the list into an array
    
    return dt_date_arr # An array of dates in datetime format

def doy_to_datetime(yr, doy_arr):
    
    '''
    This function converts an array/list of doy (day of year) values into datetime.
    
    INPUTS:
    yr: the year (int)
    doy_arr: array/list of doys (int)
    
    OUTPUTS:
    dt_date: array of dates in datetime format
    '''
    
    dt_date = []
    for doy in doy_arr:
        dt_date.append(dtt.datetime(yr, 1, 1) + dtt.timedelta(int(doy) - 1)) # Convert the doy into a date
    
    dt_date = np.array(dt_date) # Turn the list into an array
    
    return dt_date

def datetime_to_doy(dt_date_arr):
    
    '''
    This function converts an array/list of date values (in datetime format)
    into doy (day of year).
    
    INPUTS:
    dt_date_arr: array/list of dates in datetime format
    
    OUTPUTS:
    doy: array of the day of year (int)
    
    NOTES:
    More infos in: https://www.w3schools.com/python/python_datetime.asp
    '''
    
    doys = []
    for d in dt_date_arr:
        ## dt_date = dtt.datetime(yr, mnth, dy)
        doys.append(int(d.strftime('%j')))
    
    doys = np.array(doys) # Turn the list into an array
    
    return doys

def dec_mins_to_time(dec_time_arr, delimiter_par):
    
    '''
    This function converts an array/list of time values (in minutes) into full time (hour, minutes,
    seconds).
    
    INPUTS:
    dec_time_arr: array/list of time values (float)
    delimiter_par: the symbol to join hour, minutes and seconds (str, e.g. ':')
    
    OUTPUTS:
    full_time: array of full time format (str)
    '''
    
    full_time = []
    for dec_time in dec_time_arr:
    
        # First, convert the decimal time from minutes into seconds #
        s = dec_time*60
        
        # Hour #
        h = int(divmod(s, 3600)[0]) # or int(dec_time/60)
        
        # Minutes #
        m = int((s - h*3600)/60)
        
        # Seconds #
        sec = int(s - h*3600 - m*60)
        
        # Join the values #
        joined_time = delimiter_par.join([f'{h:02d}', f'{m:02d}', f'{sec:02d}'])
        full_time.append(joined_time)
    
    full_time = np.array(full_time) # Turn the list into an array
    
    return full_time

def sec_to_time(sec_arr, time_format):
    
    '''
    This function converts an array/list of time values (in seconds) into full time
    (hour, minutes and seconds) and join them into one value.
    
    INPUTS:
    sec_arr: array of time values in seconds (float)
    time_format: the format of the joined time values (str, e.g. '%H:%M:%S')
    
    OUTPUTS:
    full_time: array of time values in full time format (str)
    '''
    
    full_time = []
    for secs in sec_arr:
        temp_time = time.gmtime(secs)
        full_time.append(time.strftime(time_format, temp_time))
    
    full_time = np.array(full_time) # Turn the list into an array
    
    return full_time


''' ------ Functions for general use ------ '''

def read_AERONET_files(fl, vars_lst):
    
    '''
    This function reads the AERONET file(s) and extracts specific parameters
    (columns).
    
    INPUTS:
    fl: the AERONET file (str)
    vars_lst: list which contains the parameters we want to keep
    from the AERONET file (str)
    
    OUTPUTS:
    df: a DataFrame with the variables we choose
    '''
    
    ## Read the AERONET file and drop the last empty row(s)
    temp_df = pd.read_csv(fl, skiprows = np.arange(7))
    temp_df.drop(temp_df.tail(1).index,
                 inplace = True) # drop the last row(s), if need be
    
    ## Keep specific variables (columns) from the file ##
    df = temp_df[vars_lst]
    
    ## Join the 'Date(dd:mm:yyyy)' and 'Time(hh:mm:ss)' columns ##
    df['Date'] = temp_df[['Date(dd:mm:yyyy)', 'Time(hh:mm:ss)']]\
        .agg(' '.join, axis = 1) # works only between strings
    
    ## Turn the -999 values into Nan and drop them ##
    df = df.replace(-999, np.nan); df = df.dropna()
    
    ## Convert the dates in the 'Date' column into DateTime and set it as Index ##
    df['Date'] = pd.to_datetime(df['Date'], format = '%d:%m:%Y %H:%M:%S')
    df = df.set_index('Date')
    
    return df

def read_CAMS_files(fl, var):
    
    '''
    This function reads the CAMS (ADS or CDS) file(s) and extracts specific
    parameters (columns). This function extracts only one meteorological
    parameter from the file(s)
    
    INPUTS:
    fl: the CAMS file (str)
    var: the name of the parameter, as it is written in the file (str)
    
    OUTPUTS:
    a list which contains the parameters that were extracted from the file,
    with the following order --> [time, latitude, longitude, meteorological parameter]
    '''
    
    ## Read the file ##
    ds = nc.Dataset(fl)
    
    ## Convert the time into DateTime format ##
    tm = ds.variables['time'][:]; tm = np.array(tm)
    tm = np.array([x*3600 for x in tm]) # convert hours to seconds
    tm = utc_sec_to_datetime(tm, 31536000*(1900 - 1970) - 1468800)

    ## Keep the variable ##
    meteo_par = ds.variables[var][:]; meteo_par = np.array(meteo_par)
    
    ## Keep the latitude and the longitude ##
    lat = ds.variables['latitude'][:]; lat = np.array(lat)
    lon = ds.variables['longitude'][:]; lon = np.array(lon)
    
    return [tm, lat, lon, meteo_par]

def read_EEA_files(fls):
    
    '''
    This function reads the European Environment Agency (EEA) file(s) and
    extracts specific parameters (columns).
    
    INPUTS:
    fls: list of the EEA files (str)
    
    OUTPUTS:
    df: a DataFrame with the variables we choose
    
    NOTES:
    Some commands might need change, depending the pollutant we have.
    '''
    
    ## Keep specific columns from the file(s) ##
    df_cols = ['DatetimeBegin', 'Concentration', 'Validity', 'Verification']
    
    ## Join all the data from the files ##
    for i in range(len(fls)):
        
        # Read the file #
        temp_data = pd.read_csv(fls[i], delimiter = ',') # hourly concentrations
        
        if i == 0:
            df = temp_data[df_cols]
        
        else:
            temp_df = temp_data[df_cols]
            df = pd.concat([df, temp_df])
    
    # Convert the raw dates ('DatetimeBegin' column) into DateTime #
    df['DatetimeBegin'] = pd.to_datetime(df['DatetimeBegin'],
                                         format = '%Y-%m-%d %H:%M:%S +01:00')
    
    # Rename the 'DatetimeBegin' column #
    df.rename(columns = {'DatetimeBegin':'Date'}, inplace = True)
    # , 'Concentration':'Concentration (μg/m3)' # NO NEED!!
    
    # Sort the data by the 'Date' column and set it as Index #
    df = df.sort_values(by = ['Date'])
    df = df.set_index('Date')

    # Filter the data #
    df = df[(df['Validity'] == 1) & (df['Verification'] == 1)]
    df = df.dropna()
    
    return df

def make_2d_features(arr):
    
    '''
    This function makes a 2D array, which can be used as a features matrix for
    a Machine Learning model.
    
    INPUTS:
    arr: an array with values (int/float/str)
    
    OUTPUTS:
    newarr: the 2D features array
    '''
    
    if arr.ndim == 1:
        newarr = arr.reshape(-1, 1)
    else:
        newarr = arr
    
    return newarr

def make_figure(fg, axs, ax_choice, xplot, yplot, xy_labels):
    
    '''
    This function makes a figure with multiple plots.
    
    INPUTS:
    fg: the figure
    axs: the axes (dimensions) of the figure
    ax_choice: the parameter to choose the axes of the figure (str)
    xplot, yplot: the x and y parameters for the plot
    xy_labels: list with the labels for the x (xy_labels[0]) and y
    (xy_labels[1]) axis (str)
    
    OUTPUTS:
    the figure with the multiple plots
    
    NOTES:
    Many commands/inputs can change/comment in this function, depending what kind
    of plot(s) we want to make.
    '''
    
    ## Define the axes by hand, depending the plots ##
    # if ax_choice == 'ATHENS-NOA':
    #     ax = axs[0, 0]
    # elif ax_choice == 'ATHENS_NTUA':
    #     ax = axs[0, 1]
    # elif ax_choice == 'Finokalia-FKL':
    #     ax = axs[1, 0]
    # else:
    #     ax = axs[1, 1]
    
    if ax_choice == 'Athens':
        ax = axs[0, 0]
    elif ax_choice == 'Patra':
        ax = axs[0, 1]
    elif ax_choice == 'Volos':
        ax = axs[1, 0]
    else:
        ax = axs[1, 1]
    
    ax.scatter(xplot, yplot, label = ax_choice)
    # pax.set_xticklabels(xplot, rotation = 45)
    # ax.set_xlim([])
    ax.set_ylim([0, 900])
    ax.set_xlabel(xy_labels[0], fontsize = 15)
    ax.set_ylabel(xy_labels[1], fontsize = 15)
    ax.xaxis.grid(visible = True, which = 'major', linestyle = '--', color = 'grey')
    ax.yaxis.grid(visible = True, which = 'major', linestyle = '--', color = 'grey')
    ax.legend(fontsize = 20)

def s5p_overpass(fls, list_val, product_nm):
    
    '''
    This function makes overpass data, over a specific area (latitude,
    longitude and cirlce radius) for several S5P/TROPOMI products.
    
    INPUTS:
    fls: the S5P files (str, array/list)
    list_val: list wich contains specific values depending the product (see
    more info below)
    product_nm: the name of the product (str, valid values: 'RA_OFFL',
    'IR_OFFL', 'TO3_NRTI', 'CLOUD_OFFL', 'CH4_OFFL', etc.).
    We can add any other product name we want.
    
    OUTPUTS:
    averageproduct: the S5P overpass data
    
    NOTES:
    In this function we can add any other product we want. The product_nm though,
    must match with the values in the if/elif conditions.
    '''
    
    ## A message in case the average Harp product is not created ##
    # error_note = 'Error!! Something is wrong!!!'
    
    if product_nm == 'RA_OFFL':
        
        '''
        For the L1B_RA_OFFL file(s), the list_val parameter takes the following
        values:
        lat_val, lon_val: the latitude and longitude of a place/area (int/float)
        point_radius: the circle radius around the place/area (int/float)
        bnd: the wavelength band (int)
        '''
        
        ## Take the values from the list_val parameter ##
        lat_val, lon_val = list_val[0], list_val[1]
        point_radius, bnd = list_val[2], list_val[3]
        
        ## Define the Harp operations ##
        operations = ";".join([f'point_distance({lat_val}, {lon_val}, '+
                               f'{point_radius}[km])',
                               'keep(datetime, latitude, longitude, '+
                               'latitude_bounds, longitude_bounds, '+
                               'wavelength, photon_radiance, '+
                               'scan_subindex, index)',
                               'derive(datetime {time} [seconds since 2010-01-01])'])

        ## Define the paths in the file(s) ##
        main_path = f'/BAND{bnd}_RADIANCE/STANDARD_MODE/'
        obs_path = main_path+'OBSERVATIONS/'
        # inst_path = main_path+'INSTRUMENT/'
        # geo_path = main_path+'GEODATA/'
        
        ## Keep the variables from the files ##
        productlist = []
        for file in fls:
            
            try:
            
                ## Get the variables from the Harp ingestions ##
                product = harp.import_product(file, operations)
                    
                ## Add extra variables which are not in the Harp ingestions ##
                with coda.open(file) as pf:
                        
                    # OBSERVATIONS path #
                    temp_var = coda.fetch(pf, obs_path+'radiance_error')[0] # & 0b11 == 0)
                    temp_var = temp_var.reshape(-1, temp_var.shape[-1])
                    product.photon_radiance_error =\
                        harp.Variable(temp_var[product.index.data, :],
                                      ['time', 'spectral'])
                        
                    temp_var = coda.fetch(pf, obs_path+'radiance_noise')[0] # & 0b11 == 0)
                    temp_var = temp_var.reshape(-1, temp_var.shape[-1])
                    product.photon_radiance_noise =\
                        harp.Variable(temp_var[product.index.data, :],
                                      ['time', 'spectral'])
                        
                    temp_var = coda.fetch(pf, obs_path+'ground_pixel_quality')[0] # & 0b11 == 0)
                    temp_var = temp_var.flatten()
                    product.ground_pixel_quality =\
                        harp.Variable(temp_var[product.index.data], ['time'])
                        
                    temp_var = coda.fetch(pf, obs_path+'quality_level')[0] # & 0b11 == 0)
                    temp_var = temp_var.reshape(-1, temp_var.shape[-1])
                    product.quality_level =\
                        harp.Variable(temp_var[product.index.data, :],
                                      ['time', 'spectral'])
                        
                    temp_var = coda.fetch(pf, obs_path+'spectral_channel_quality')[0] # & 0b11 == 0)
                    temp_var = temp_var.reshape(-1, temp_var.shape[-1])
                    product.spectral_channel_quality =\
                        harp.Variable(temp_var[product.index.data, :],
                                      ['time', 'spectral'])
                        
                    # GEODATA path #
                    # temp_var = coda.fetch(pf, geo_path+'viewing_azimuth_angle')[0]
                    # temp_var = temp_var.flatten()
                    # product.viewing_azimuth_angle =\
                    #     harp.Variable(temp_var[product.index.data], ['time'])
                        
                    # temp_var = coda.fetch(pf, geo_path+'viewing_zenith_angle')[0]
                    # temp_var = temp_var.flatten()
                    # product.viewing_zenith_angle =\
                    #     harp.Variable(temp_var[product.index.data], ['time'])
            
                    productlist.append(product)
            
            except:
                pass
                
        ## Make the average product ##
        # try:
        averageproduct = harp.execute_operations(productlist)
        return averageproduct
        # except:
        #     return error_note
    
    elif product_nm == 'IR_OFFL':
        
        '''
        For the L1B_IR_OFFL file(s), the list_val parameter takes the following
        values:
        bnd: the wavelength band (int)
        
        Note that the L1_IR file(s) DO NOT include lat and lon information. So,
        in this case we just keep specific data from the file(s). NO overpass
        file(s) is (are) created.
        '''
        
        ## Take the values from the list_val parameter ##
        bnd = list_val[0]
        
        ## Define the Harp operations ##
        operations = ";".join(['keep(datetime, wavelength, photon_irradiance, '+
                               'scan_subindex, index)',
                               'derive(datetime {time} [seconds since 2010-01-01])'])

        ## Define the paths in the file(s) ##
        # main_path = f'/BAND{bnd}_IRRADIANCE/STANDARD_MODE/'
        obs_path = f'/BAND{bnd}_IRRADIANCE/STANDARD_MODE/OBSERVATIONS/'
        # inst_path = main_path+'INSTRUMENT/'
        
        ## Keep the variables from the files ##
        productlist = []
        for file in fls:
            
            try:
            
                ## Get the variables from the Harp ingestions ##
                product = harp.import_product(file, operations, f'band = {bnd}')
                    
                ## Add extra variables which are not in the Harp ingestions ##
                with coda.open(file) as pf:
                        
                    # OBSERVATIONS path #
                    temp_var = coda.fetch(pf, obs_path+'irradiance_error')[0] # & 0b11 == 0)
                    temp_var = temp_var.reshape(-1, temp_var.shape[-1])
                    product.photon_irradiance_error =\
                        harp.Variable(temp_var[product.index.data, :],
                                      ['time', 'spectral'])
                        
                    temp_var = coda.fetch(pf, obs_path+'irradiance_noise')[0] # & 0b11 == 0)
                    temp_var = temp_var.reshape(-1, temp_var.shape[-1])
                    product.photon_irradiance_noise =\
                        harp.Variable(temp_var[product.index.data, :],
                                      ['time', 'spectral'])
                        
                    temp_var = coda.fetch(pf, obs_path+'quality_level')[0] # & 0b11 == 0)
                    temp_var = temp_var.reshape(-1, temp_var.shape[-1])
                    product.quality_level =\
                        harp.Variable(temp_var[product.index.data, :],
                                      ['time', 'spectral'])
                        
                    temp_var = coda.fetch(pf, obs_path+'spectral_channel_quality')[0] # & 0b11 == 0)
                    temp_var = temp_var.reshape(-1, temp_var.shape[-1])
                    product.spectral_channel_quality =\
                        harp.Variable(temp_var[product.index.data, :],
                                      ['time', 'spectral'])
                        
                    productlist.append(product)
            
            except:
                pass
                
        ## Make the average product ##
        # try:
        averageproduct = harp.execute_operations(productlist)
        return averageproduct
        # except:
        #     return error_note
    
    elif product_nm == 'TO3_NRTI':
        
        '''
        For the O3_NRTI (Total O3) file(s), the list_val parameter take the
        following values:
        lat_val, lon_val: the latitude and longitude of a place/area (int/float)
        point_radius: the circle radius around the place/area (int/float)
        qa_flag: the qa flag threshold (int)
        '''
        
        ## Take the values from the list_val parameter ##
        lat_val, lon_val = list_val[0], list_val[1]
        point_radius, qa_flag = list_val[2], list_val[3]
        
        ## Keep the fuctor to convert the units into DU ##
        temp_file = nc.Dataset(fls[0])
        conv_val = temp_file['PRODUCT/ozone_total_vertical_column'].\
            multiplication_factor_to_convert_to_DU # from mol/m2 to DU
        
        ## Define the Harp operations ##
        operations = ";".join([f'point_distance({lat_val}, {lon_val}, '+
                               f'{point_radius}[km])',
                               f'O3_column_number_density_validity > {qa_flag}',
                               'keep(datetime_start, latitude, longitude, '+
                               'latitude_bounds, longitude_bounds, '+
                               'solar_zenith_angle, solar_azimuth_angle, '+
                               'surface_altitude, surface_pressure, '+
                               'scan_subindex, index)',
                               'derive(datetime_start {time} [seconds since 2010-01-01])'])

        operations2 = ';'.join(['keep(O3_column_number_density)'])

        ## Define the paths in the file(s) ##
        # meas_path = '/PRODUCT/'
        input_path = '/PRODUCT/SUPPORT_DATA/INPUT_DATA/'
        geo_path = '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/'
        det_path = '/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/'
        
        ## Keep the variables from the files ##
        productlist = []
        for file in fls:
            
            try:
            
                ## Get the variables from the Harp ingestions ##
                product = harp.import_product(file, operations)
                
                ## Change the units in the ozone column and keep the values ##
                product2 = harp.import_product(file, operations2)
                temp_var = product2.O3_column_number_density.data*conv_val # 2241.15
                product.ozone_total_vertical_column =\
                    harp.Variable(temp_var[product.index.data], ['time'])
                product.ozone_total_vertical_column.unit = 'DU'
                
                ## Add extra variables which are not in the Harp ingestions ##
                with coda.open(file) as pf:
                
                    # GEOLOCATIONS path #
                    temp_var = coda.fetch(pf, geo_path+'viewing_azimuth_angle')[0]
                    temp_var = temp_var.flatten()
                    product.viewing_azimuth_angle =\
                        harp.Variable(temp_var[product.index.data], ['time'])
                        
                    temp_var = coda.fetch(pf, geo_path+'viewing_zenith_angle')[0]
                    temp_var = temp_var.flatten()
                    product.viewing_zenith_angle =\
                        harp.Variable(temp_var[product.index.data], ['time'])
                        
                    # INPUT_DATA path #
                    temp_var = coda.fetch(pf, input_path+'surface_temperature')[0]
                    temp_var = temp_var.flatten()
                    product.surface_temperature =\
                        harp.Variable(temp_var[product.index.data], ['time'])
                        
                    ''' For the O3 OFFL files '''
                    # temp_var = coda.fetch(pf, input_path+'cloud_fraction_crb')[0]
                    # temp_var = temp_var.flatten()
                    # product.cloud_fraction_crb =\
                    #     harp.Variable(temp_var[product.index.data], ['time'])
                        
                    # temp_var = coda.fetch(pf, input_path+'cloud_albedo_crb')[0]
                    # temp_var = temp_var.flatten()
                    # product.cloud_albedo_crb =\
                    #     harp.Variable(temp_var[product.index.data], ['time'])
                        
                    # temp_var = coda.fetch(pf, input_path+'cloud_height_crb')[0]
                    # temp_var = temp_var.flatten()
                    # product.cloud_height_crb =\
                    #     harp.Variable(temp_var[product.index.data], ['time'])
                    ''' --------------------- '''
                        
                    # DETAILED_RESULTS path #
                    temp_var = coda.fetch(pf, det_path+'effective_scene_albedo')[0]
                    temp_var = temp_var.flatten()
                    product.effective_scene_albedo =\
                        harp.Variable(temp_var[product.index.data], ['time'])
                        
                    productlist.append(product)
            
            except:
                pass

        ## Make the average product ##
        # try:
        averageproduct = harp.execute_operations(productlist)
        return averageproduct
        # except:
        #     return error_note
    
    elif product_nm == 'CLOUD_OFFL':
        
        '''
        For the L2_CLOUD_OFFL file(s), the list_val parameter take the following
        values:
        lat_val, lon_val: the latitude and longitude of a place/area (int/float)
        point_radius: the circle radius around the place/area (int/float)
        qa_flag: the qa flag threshold (int)
        model_nm: the name of the model (str, CAL or CRB)
        bnd: the wavelength band (str, UVVIS or NIR) --> GIVES AN ERROR!!!
        
        Note that depending the model we want (CAL or CRB), the operations for
        the harp product change.
        '''
        
        ## Take the values from the list_val parameter ##
        lat_val, lon_val = list_val[0], list_val[1]
        point_radius, qa_flag = list_val[2], list_val[3]
        model_nm, bnd = list_val[4], list_val[5]
        
        ## Define the Harp operations and make the average product ##
        if model_nm == 'CAL':
            
            operations = ";".join([f'point_distance({lat_val}, {lon_val}, '+
                                   f'{point_radius}[km])',
                                   f'cloud_fraction_validity > {qa_flag}',
                                   'keep(datetime_start, latitude, longitude, '+
                                   'latitude_bounds, longitude_bounds, '+
                                   'cloud_fraction, cloud_base_height, '+
                                   'cloud_base_height_uncertainty, '+
                                   'cloud_top_height, cloud_top_height_uncertainty, '+
                                   'scan_subindex, index)',
                                   'derive(datetime_start {time} '+
                                   '[seconds since 2010-01-01])'])
        
        else:
            
            operations = ";".join([f'point_distance({lat_val}, {lon_val}, '+
                                   f'{point_radius}[km])',
                                   f'cloud_fraction_validity > {qa_flag}',
                                   'keep(datetime_start, latitude, longitude, '+
                                   'latitude_bounds, longitude_bounds, '+
                                   'cloud_fraction, cloud_albedo, cloud_height, '+
                                   'cloud_fraction_uncertainty, '+
                                   'cloud_albedo_uncertainty, '+
                                   'cloud_height_uncertainty, '+
                                   'scan_subindex, index)',
                                   'derive(datetime_start {time} '+
                                   '[seconds since 2010-01-01])'])
        
        ## Make the average product ##
        # try:
        averageproduct = harp.import_product(fls, operations,
                                             f'model = {model_nm}')
        # , f'band = {bnd}'
        return averageproduct
        # except:
        #     return error_note
    
    elif product_nm == 'CH4_OFFL':
        
        '''
        For the L2_CH4_OFFL file(s), the list_val parameter take the following
        values:
        lat_val, lon_val: the latitude and longitude of a place/area (int/float)
        point_radius: the circle radius around the place/area (int/float)
        qa_flag: the qa flag threshold (int)
        '''
        
        ## Take the values from the list_val parameter ##
        lat_val, lon_val = list_val[0], list_val[1]
        point_radius, qa_flag = list_val[2], list_val[3]
        
        ## Define the Harp operations ##
        operations = ";".join([f'point_distance({lat_val}, {lon_val}, '+
                               f'{point_radius}[km])',
                               'CH4_column_volume_mixing_ratio_dry_air_validity '+
                               f'> {qa_flag}',
                               'keep(datetime_start, latitude, longitude, '+
                               'latitude_bounds, longitude_bounds, '+
                               'surface_meridional_wind_velocity, '+
                               'surface_zonal_wind_velocity, '+
                               'H2O_column_number_density, '+
                               'H2O_column_number_density_uncertainty, '+
                               'scan_subindex, index)',
                               'derive(datetime_start {time} [seconds since 2010-01-01])'])
        
        ## Make the average product ##
        # try:
        averageproduct = harp.import_product(fls, operations)
        return averageproduct
        # except:
        #     return error_note
    
    else: # for PAL TCWV
        
        '''
        For the L2_TCWV_PAL file(s), the list_val parameter take the following
        values:
        lat_val, lon_val: the latitude and longitude of a place/area (int/float)
        point_radius: the circle radius around the place/area (int/float)
        qa_flag: the qa flag threshold (int)
        '''
        
        ## Take the values from the list_val parameter ##
        lat_val, lon_val = list_val[0], list_val[1]
        point_radius, qa_flag = list_val[2], list_val[3]
        
        ## Define the Harp operations ##
        operations = ";".join([f'point_distance({lat_val}, {lon_val}, '+
                               f'{point_radius}[km])',
                               'water_vapor_column_density_validity '+
                               f'> {qa_flag}',
                               'keep(datetime_start, latitude, longitude, '+
                               'latitude_bounds, longitude_bounds, '+
                               'water_vapor_column_density, '+
                               'water_vapor_column_density_uncertainty, '+
                               'water_vapor_column_density_amf, '+
                               'water_vapor_column_density_avk, '+
                               'scan_subindex, index)',
                               'derive(datetime_start {time} [seconds since 2010-01-01])'])
        
        ## Make the average product ## --> NOT WORKING!!!!
        # try:
        averageproduct = harp.import_product(fls, operations)
        return averageproduct
        # except:
        #     return error_note

def remove_nan(x_arr, y_arr):
    
    '''
    This function removes the Nan values of two arrays and keeps only the common
    (no Nan) ones. The result is that x_arr and y_arr have the same length and not
    any Nan values.
    
    INPUTS:
    x_arr, y_arr: arrays (int/float)
    
    OUTPUTS:
    x_arr, y_arr: arrays (int/float) with the same length and not any Nan values
    
    NOTES:
    This step is a prior step when we want to apply a fitting curve in x_arr and
    y_arr (e.g. the next two functions), in order x_arr and y_arr to have the same
    length and not any Nan values. If x_arr and y_arr do not have any Nan values are
    in the first place, then these two arrays will remain as they are and we do not
    need to use this function. With a few modifications this function can be used for
    more than two variables, as:
    
    def remove_nan(x_arr, y_arr, z_arr, ...):
        x_arr = np.array(x_arr); y_arr = np.array(y_arr), z_arr = np.array(z_arr), ...
        filter_condition = np.logical_and(~np.isnan(x_arr), np.isnan(y_arr),
                                          np.isnan(z_arr), ...)
        x_arr = x_arr[np.where(filter_condition)]
        y_arr = y_arr[np.where(filter_condition)]
        z_arr = y_arr[np.where(filter_condition)]
        ...
        
        return x_arr, y_arr, z_arr, ...
    '''
    
    ## Convert the input variables into arrays, even they already are ##
    x_arr = np.array(x_arr); y_arr = np.array(y_arr)
    
    ## Keep the common values (the no Nan values) between x_arr and y_arr ##
    filter_condition = np.logical_and(~np.isnan(x_arr), ~np.isnan(y_arr))
    x_arr = x_arr[np.where(filter_condition)]
    y_arr = y_arr[np.where(filter_condition)]
    
    return x_arr, y_arr

def poly_fit(x, y, polyfit_order):
    
    '''
    This function fits a linear (or polynomial) curve in two variables. Nan values
    must NOT be included in the input arrays.
    
    INPUTS:
    x: array of the independent variable x in the funtion (int/float)
    y: array of the dependent variable y in function (int/float)
    polyfit_order: the order of the polynomial (int)
    
    OUTPUTS:
    model: array/list of the coefficients of the polynomial curve (float)
    poly_curve: array/list of the values of the the polynomial function (float)
    R: correlation coefficient R^2 (float)
    
    NOTES:
    1) The model[i] value must be the same with the order of the polynomial. For
    example if we have a 3rd degree polynomial, model[3] gives the coefficient of
    the x^3. Same the rest. We use the model[i] value in lables, in .plot() commands,
    as follows: label = ...function (str)... .format(mymodel[...], mymodel[...],),
    depending the degree of the curve. For instance, for a 2nd degree polynomial we do:
    label = 'y = {:.3f}x$^2$ + ({:.3f})x + ({:.3f}) \n R$^2$ = {:.3f}'.format(mymodel[2],
    mymodel[1], mymodel[0], R).

    2) If we do not use the np.poly1d(), R cannot be calculated automatically.
    '''
    
    ## Convert the input variables into arrays, even they already are ##
    x = np.array(x); y = np.array(y)
    
    ## Keep the common values (no Nan values) between x and y - Apply remove_nan
    ## function
    x_new, y_new = remove_nan(x, y)
    
    ## Make the linear/polynomial fit ##
    model = np.poly1d(np.polyfit(x_new, y_new, polyfit_order))
    R = r2_score(y_new, model(x_new))
    poly_curve = np.polyval(model, x_new)
    
    return poly_curve, model, R

def exp_fit(x, y, fit_fun):
    
    '''
    This function fits an exponential curve in two variables. Nan values must
    NOT be included in the input arrays.
    
    INPUTS:
    x: array of the independent variable x in the exponential equation (int/float)
    y: array of the dependent variable y in the exponential equation (int/float)
    fit_fun: the type of the exponential function (int, 1: y = be^(ax), 2: y = bx^(a))
    
    OUTPUTS:
    model: array/list of the coefficients of the exponential equation (float)
    exp_curve: array/list of the values of the the exponential function (float)
    '''
    
    ## Convert the input variables into arrays, even they already are ##
    x = np.array(x); y = np.array(y)
    
    ## Keep the common values (no Nan values) between x and y - Apply remove_nan
    ## function
    x_new, y_new = remove_nan(x, y)
    
    ## Define the exponential curve and make the fit ##
    if fit_fun == 1:
        model = np.poly1d(np.polyfit(x_new, np.log(y_new), 1, w = np.sqrt(y_new)))
        exp_curve = np.exp(model[0])*np.exp(model[1]*x_new)
    
    else:
        model = np.poly1d(np.polyfit(np.log10(x_new), np.log10(y_new), 1))
        exp_curve = 10**model[0]*(x_new**model[1])
    
    return exp_curve, model

def multi_col_df(data_list, index_arr, main_cols, secondary_cols):
    
    '''
    This function gives as a result a multi-column DataFrame (based on the master thesis
    analysis).
    
    INPUTS:
    data_list: list which contains the variables/data we want to insert to the DataFrame
    index_arr: array/list of the indexes of the DataFrame (int, float, datetime, etc.)
    main_cols: array/list of the main columns of the DataFrame (str)
    secondary_cols: array/list of the secondary columns of the DataFrame (str)
    
    OUTPUTS:
    df: the multi-column DataFrame
    '''
    
    df = pd.DataFrame()
    for var in range(len(data_list)):
                
        if var == 0:
            
            # Define the DataFrame at the first iteration and make the multi-columns #
            df = pd.DataFrame(data_list[var], index = index_arr, columns = secondary_cols)
            df.columns = pd.MultiIndex.from_product([[main_cols[var]], df.columns])
        
    else:
        
        # Add the other varibles to the DataFrame as multi-column format #
        df = df.join(pd.DataFrame(data_list[var], index = df.index,
                                  columns = pd.MultiIndex.from_product([[main_cols[var]], secondary_cols])))
    
    return df

def get_lat_lon_number(latmin, latmax, lonmin, lonmax, gr_deg):
    
    '''
    This function divides a range of latitude and longitude by a given grid degree and
    returns the number of the latitudes and longitudes fits in that range.
    
    INPUTS:
    latmin, latmax: minimum and maximum value of the latitude range, respectively (int/float)
    lonmin, lonmax: minimum and maximum value of the longitude range, respectively (int/float)
    gr_deg: the degree of the grid (float)
    
    OUTPUTS:
    lat_num, lon_num: the number of the latitudes and the longitudes fit in the range (int)
    '''
    
    lat_num = int((latmax - (latmin))/gr_deg) + 1
    lon_num = int((lonmax - (lonmin))/gr_deg) + 1
    
    return lat_num, lon_num

def make_lv3_data(lat_val, lon_val, lon_n, lat_n, arr):
    
    '''
    This function converts L2 satellite data to L3 (gridded). Nan values must NOT be included in
    the input arrays.
    
    INPUTS:
    lat_val, lon_val: arrays of latitude and longitude, respectively (int/float)
    lon_n, lat_n: the number of the longitudes and the latitudes (derived from the get_lat_lon_number
    function, int)
    arr: array of the L2 product to be gridded (int/float)
    
    OUTPUTS:
    zdata: the gridded L2 product (2D, int/float)
    ydata: array of latitude for the gridded data (int/float)
    xdata: array of longitude for the gridded data (int/float)
    '''
    
    zdata, ydata, xdata = np.histogram2d(lat_val, lon_val, bins = (lat_n-1, lon_n-1),
                                         weights = arr) # , normed = False
    counts, _, _ = np.histogram2d(lat_val, lon_val, bins = (lat_n-1, lon_n-1))
    zdata = zdata/counts
    
    return zdata, ydata, xdata

def make_lv2_maps(x, y, z, lonmin, lonmax, latmin, latmax, vmin, vmax,
                  map_projection, colorscale, plot_units, figtitle):
    
    '''
    This function plots a L2 satellite product in a map projection. Nan values must NOT be included in
    the input arrays.
    
    INPUTS:
    x: array of longitude (int/float)
    y: array of latitude (int/float)
    z: array of the L2 product to be visualized (int/float)
    lonmin, lonmax: minimum and maximum values of longitude (the range of the longitude in the map,
    int/float)
    latmin, latmax: minimum and maximum values of latitude (the range of the latitude in the map,
    int/float)
    vmin, vmax: minimum and maximum values of the L2 product (the range of the colorscale, int/float)
    map_projection: the projection of the map (e.g. ccrs.PlateCarree())
    colorscale: the colorscale of the plot (str)
    plot_units: the units of the L2 product (str)
    figtitle: the title of the plot (str)
    
    OUTPUTS:
    the map of the L2 product in a specific area
    '''
    
    fig_size = (20, 10); label_size = 14
    edge_color = 'black'; line_width = 1; line_style = '--'
    map_orientation = 'horizontal'
    s_par = 1; fraction_par = 0.04; pad_par = 0.1
    
    # Make the plot #
    fig = plt.figure(figsize = fig_size)
    ax = plt.axes(projection = map_projection)
    ax.set_extent([lonmin, lonmax, latmin, latmax], map_projection)

    img = plt.scatter(x, y, c = z,
                      vmin = vmin, vmax = vmax,
                      cmap = colorscale,
                      s = s_par,
                      transform = map_projection)
       
    ax.coastlines()
    fig_title = figtitle
    
    ax.add_feature(cfeature.BORDERS,
                   edgecolor = edge_color, linewidth = line_width)
    ax.add_feature(cfeature.COASTLINE,
                   edgecolor = edge_color, linewidth = line_width)

    gl = ax.gridlines(draw_labels = True, linestyle = line_style)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size':label_size}
    gl.ylabel_style = {'size':label_size}

    cbar = fig.colorbar(img, ax = ax,
                        orientation = map_orientation,
                        fraction = fraction_par, pad = pad_par)
    cbar.set_label(plot_units, fontsize = 16)
    cbar.ax.tick_params(labelsize = label_size)
    plt.title(fig_title, fontsize = 20)

def scattering_angle(sza_ang, vza_ang, vaa_ang, saa_ang):
    
    '''
    This function calculates the scattering angle.
    
    INPUTS:
    sza_ang: array of the solar zenith angle values (float)
    vza_ang: array of the viewing zenith angle values (float)
    vaa_ang: array of the viewing azimuth angle values (float)
    saa_ang: array of the solar azimuth angle values (float)
    
    OUTPUTS:
    sca_ang: array of the scattering angle values (float)
    
    NOTES:
    An alternative formula is the following:
    
    par1 = -np.cos(np.radians(vza_ang))*np.cos(np.radians(sza_ang))
    par2 = np.sqrt((1 - np.power(np.cos(np.radians(vza_ang)), 2))*
                   (1 - np.power(np.cos(np.radians(sza_ang)), 2)))*
                    np.cos(np.radians(vaa_ang - saa_ang))
    sca = 180 - np.degrees(np.arccos(par1 + par2))
    '''
    
    par1 = np.cos(np.radians(sza_ang))*np.cos(np.radians(vza_ang))
    par2 = np.sin(np.radians(sza_ang))*np.sin(np.radians(vza_ang))*\
        np.cos(np.radians(vaa_ang - saa_ang))
    sca_ang = np.degrees(np.arccos(par1 - par2))
    
    return sca_ang

def consecutive(arr, par_list):
    
    '''
    This function checks if the elements of an array/list are consecutive.
    
    INPUTS:
    arr: the array/list to be checked (int/float)
    par_list: list which contains specific values
    1) if arr contains int values, par_list = [step_val] where step_val is the
    step in the array/list (int/float),
    2) if arr contains float values, par_list = [step_val, decs_val], where
    step_val is the same as before and decs_val is the number of the decimals
    
    OUTPUTS:
    True, if the elements are consecutive, False, if they are not (bool)
    '''
    
    ## Convert the 'arr' into an array, even if already is, sort it and get its type ##
    arr = np.array(arr); arr.sort(); arr_type = arr.dtype
    
    ## Get the step for the array ##
    step_val = par_list[0]
    
    ## Check if the elements are consecutive ##
    if 'float' in [arr_type]:
        
        decs_val = par_list[1]
        
        for i in range (1, len(arr)):
            
            if(arr[i] != round(arr[i-1]+step_val, decs_val)):
                return False
    
    else:
        
        for i in range (1, len(arr)):
            
            if(arr[i] != arr[i-1]+step_val):
                return False
    
    return True

def area_of_interest(lat_val, lon_val, coords_step):
    
    '''
    This function makes a (square/rhombus) area of interest for specific
    coordinates.
    
    INPUTS:
    lat_val, lon_val: the latitude and the longitude (int/float)
    coords_step: the step for the coordinates (int/float)
    
    OUTPUTS:
    aoi: list in format [north edge, west edge, south edge, east edge] (int/float)
    '''
    
    aoi = [lat_val+coords_step, lon_val-coords_step,
           lat_val-coords_step, lon_val+coords_step]
    
    return aoi

def temperature_conversion(vals, conv_val):
    
    '''
    This function can be used to convert a temperature values into different
    temperature units. The valid conversions are: Celsius to Kelvin and vice
    versa, Celsius to Fahrenheit and vice versa and Kelvin to Fahrenheit and
    vice versa.
    
    INPUTS:
    vals: the temperature value(s) (int/float, list/array or just one value)
    conv_val: the type of the conversion (str, valid values: 'C2K', 'K2C', 'C2F',
    'F2C', 'K2F', 'F2K')
    
    OUTPUTS:
    temperature: the temperature value(s) in the new units
    
    NOTES:
    In this function, we can add any other conversion we want.
    '''
    
    if conv_val not in ['C2K', 'K2C', 'C2F', 'F2C', 'K2F', 'F2K']:
        return 'Error!! Invalid conversion!!'
    
    else:
        
        if conv_val == 'C2K':
            temperature = 273.15 + vals
        
        elif conv_val == 'K2C':
            temperature = vals - 273.15
        
        elif conv_val == 'F2C':
            temperature = (vals - 32)/1.8
        
        elif conv_val == 'C2F':
            temperature = 1.8*vals + 32
        
        elif conv_val == 'K2F':
            temperature = vals*9/5 - 459.67
        
        else:
            temperature = (vals + 459.67)*5/9
        
        return temperature

