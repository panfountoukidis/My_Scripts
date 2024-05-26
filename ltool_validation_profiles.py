# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:43:13 2022

@author: lenovo-pc
"""

import os
import glob
import numpy
import sys 
from pylab import savefig
from matplotlib import pyplot as plt
import math
import time
from netCDF4 import Dataset
import netCDF4
import xarray as xr
from pylab import polyfit
from scipy.stats.stats import pearsonr
from scipy import interpolate
from scipy import stats
import xlrd
import numpy as np
import calendar
from dateutil.parser import parse 
from scipy.interpolate import interp1d
import pandas as pd
import datetime
import datetime as dtt
import xlrd
from datetime import timedelta, date
from matplotlib import cm
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pylab
import pytest
from matplotlib.font_manager import FontProperties


#### Get the LTOOL and OPTICAL main path
main_path = 'C:/Users/lenovo-pc/Documents/METAPTIXIAKO/DIPLOMATIKI/Data_In_Out/'
main_path = 'C:/Users/lenovo-pc/Documents/Data_In_Out/' # Test!!

#### Import functions from the 'ltool_validation_functions' script
sys.path.append(main_path+'ltools_v2.0/functions/')
from ltool_validation_functions import get_ltool_data, get_bsc_or_ext_data, sort_str_wls,\
    make_figure, make_single_plot, get_rows_columns



#### Save the stations and the dilation values - Not necessary!!!
# stations = []
# dilation_vals = []


ltool_dir_main = os.path.join(main_path, 'ltools_v2.0/input/SCC/LTOOL_products/')
optical_dir_main = os.path.join(main_path, 'ltools_v2.0/input/SCC/OPTICAL_products_New_Format/')
dir_out = os.path.join(main_path, 'ltools_v2.0/output/')

dpi_val = 200
dict_title_sep = ' | '


sys_exit_parameter = 'Thessaloniki'

#### Get the LTOOL folders
# st_id = 'po04'
ltool_file_path = os.path.join(ltool_dir_main, 'Potenza/2021/*/*/') # 20220817{st_id}
ltool_folders = glob.glob(ltool_file_path); ltool_folders.sort()


#### The LTOOL variables for the Dataframe
ltool_cols = ['Base', 'Top', 'Center of mass']
ltool_var_names = ['base', 'top', 'center_of_mass']
ltool_dict = {}
res_layer_dict = {}


#### The OPTICAL variables for the Dataframe
# optical_cols = ['Backscatter', 'Extinction', 'Altitude']
# optical_var_names = ['backscatter_coef', 'extinction_coef', 'altitude'] # Temp (for main code)
optical_dict = {}


#### The OPTICAL properties variables for the Dataframe
# optical_prop_cols = ['LR', 'BAE', 'AE', 'Altitude']
# optical_prop_var_names = ['lr', 'bae', 'ae', 'altitude']
# optical_prop_dict = {}

'''
NOTE:
1) The format of the LTOOL data folders must be the following:
.../station_name/year/day/a_dilation_value/files (e.g. .../Potenza/2021/20210402pot1931/a_200/files)
2) The format of the OPTICAL data folders must be the following:
.../station_name/year/day/day/bsc (or ext)/files (e.g. .../Potenza/2021/20210402pot1931/20210402pot1931/
bsc(or ext)/files). For this case, run the make_optical_bsc-ext_folders script first.
'''


#### Initialization - Plots' parameters and filters

## Common dates from both LTOOL and OPTICAL files, as datetime
dt_dates = []

## Plots' parameters
# font_size_label = 12
# font_size_title = 15
# font_size_annotate = 10
# font_weight_label = 'bold'
# fig_size = (13, 8)
# anchor_pos = (0.5, 1)
# ylabel_text = 'Altitude [m]'
# # font_size_ticks = 7
# color_line_355 = 'blue'
# color_line_532 = 'green'
# color_line_1064 = 'red'

## Filters for the variables
alt_min = 100; alt_max = 5000 #; alt_max_ylim = 6000
bsc_ext_min = 0; bsc_ext_max = 0.3E-5 # 10E-8 # 10E-6
min_xval, max_xval = bsc_ext_min, bsc_ext_max

# bae_ae_min = -2; bae_ae_max = 4
# lr_min = 0; lr_max = 120
convert_alt = 1000 # ; alpha = 0.3


## Position for the annotate command
# annotate_text = bsc_ext_max - (0.2*10E-8) # 0.8E-7
label_pos = 0

# sys.exit()

for folder in ltool_folders:
    
    # original (when in the main folder we have */*/*/...) #
    # station = folder.split('\\')[1]
    # yr = folder.split('\\')[2]
    # date_path = folder.split('\\')[3][0:8]
    # dilation = folder.split('\\')[4].split('_')[1]
    
    # when in the main folder we define the station, the year, etc... #
    station = folder.split('/')[-2] # .split('\\')[1]
    yr = folder.split('/')[-1].split('\\')[0]
    date_path = folder.split('/')[-1].split('\\')[1][0:8]
    dilation = folder.split('/')[-1].split('\\')[-2].split('_')[1]
    
    # for a specific date folder - NO USE!! #
    # station = folder.split('/')[-3] # .split('\\')[1]
    # yr = folder.split('/')[-2] # .split('\\')[0]
    # date_path = folder.split('/')[-1].split('\\')[0][0:8]
    # dilation = folder.split('/')[-1].split('\\')[1].split('_')[1]
    
    # stations.append(station)
    # dilation_vals.append(dilation)
    
    ## Create the figure's output path step by step
    # path_out = os.path.join(dir_out, f'{station}/{yr}/')
    # if not os.path.exists(path_out):
    #     os.mkdir(path_out)
    
    # path_out2 = os.path.join(path_out, 'Profiles/')
    # if not os.path.exists(path_out2):
    #     os.mkdir(path_out2)
    
    # path_out3 = os.path.join(path_out2, f'a_{dilation}/')
    # if not os.path.exists(path_out3):
    #     os.mkdir(path_out3)
    
    ## ------- NO NEED of the following ------- ##
    # fig_out1 = os.path.join(dir_out, station) # +'/'+yr+'/a_'+dilation
    # if not os.path.exists(fig_out1): # Create the output path, if it does not exist
    #     os.mkdir(fig_out1)
    
    # fig_out2 = os.path.join(fig_out1, yr)
    # if not os.path.exists(fig_out2): # Create the output path, if it does not exist
    #     os.mkdir(fig_out2)
    
    # day_folder = date_path
    # fig_out3 = os.path.join(fig_out2, day_folder)
    # if not os.path.exists(fig_out3): # Create the output path, if it does not exist
    #     os.mkdir(fig_out3)
    
    # ## Folder for the profiles
    # fig_out_profs = os.path.join(fig_out3, 'Profiles')
    # if not os.path.exists(fig_out_profs): # Create the output path, if it does not exist
    #     os.mkdir(fig_out_profs)
    
    ## ----------------------------------- ##
    
    ## Get the LTOOL files from a specific folder
    ltool_files_temp = os.path.join(folder, '*')
    ltool_files = glob.glob(ltool_files_temp); ltool_files.sort()
    
    ## Loop in order to extract the dates from the LTOOL files' name
    check_ltool_start_date = [] # The first date in the LTOOL file's name
    check_ltool_filename_date = [] # Both dates in the LTOOL file's name
    
    for ltool_file in ltool_files:
        
        ltool_file_start_date = os.path.splitext(ltool_file)[0].split('\\')[-1].split('_')[4]
        ltool_file_start_date_dt = dtt.datetime.strptime(ltool_file_start_date, '%Y%m%d%H%M')
        check_ltool_start_date.append(ltool_file_start_date_dt)
        
        
        ltool_filename_date = ltool_file.split('\\')[-1].split('_')[4]+'_'+\
            ltool_file.split('\\')[-1].split('_')[5]
        check_ltool_filename_date.append(ltool_filename_date)
    
    ## Get the unique dates from the above variables
    unique_ltool_start_dates_dt = list(set(check_ltool_start_date))
    unique_ltool_start_dates_dt.sort()
    unique_ltool_filename_dates = list(set(check_ltool_filename_date))
    unique_ltool_filename_dates.sort()
    
    for i in range(len(unique_ltool_start_dates_dt)):
        
        ## Check if there are OPTICAL files for each LTOOL date
        optical_files_temp = os.path.join(optical_dir_main, station+'/'+yr+'/'+date_path+'*/*/*/'+
                                          '*'+unique_ltool_filename_dates[i]+'*')
        
        # for a specific date folder #
        # optical_files_temp = os.path.join(optical_dir_main, station+'/'+yr+'/'+date_path+f'{st_id}/*/*/'+
        #                                   '*'+unique_ltool_filename_dates[i]+'*')
        
        optical_files = glob.glob(optical_files_temp); optical_files.sort()
        
        if len(optical_files) == 0:
            print('There are no optical files for', station, ', on', unique_ltool_start_dates_dt[i])
        else:
            ## Append the common dates and find the measurements' time duration
            dt_dates.append(unique_ltool_start_dates_dt[i])

            start_time = unique_ltool_filename_dates[i].split('_')[0]
            end_time = unique_ltool_filename_dates[i].split('_')[1]
        
            start_time_dt = dtt.datetime.strptime(start_time, '%Y%m%d%H%M')
            end_time_dt = dtt.datetime.strptime(end_time, '%Y%m%d%H%M')
        
            time_duration = str(unique_ltool_start_dates_dt[i].date())+' '+\
                str(start_time_dt.time())+'-'+str(end_time_dt.time())
        
            ## Make the key parameter for the dictionaries
            dict_key = station+dict_title_sep+str(unique_ltool_start_dates_dt[i])+\
                dict_title_sep+'a = '+dilation
            
            
            ## --------------- 1) Get the LTOOL data --------------- ##
        
            ## LTOOL files for a specific date
            new_ltool_files_temp = os.path.join(folder, '*'+unique_ltool_filename_dates[i]+'*')
            new_ltool_files = glob.glob(new_ltool_files_temp); new_ltool_files.sort()
            
            #### Get the LTOOL data - New approach ####
            
            rows, columns, ltool_wls = get_rows_columns(new_ltool_files)
            
            base, top, center_of_mass, ltool_layers_cols, residual_values = \
                get_ltool_data(new_ltool_files, rows, columns, ltool_var_names, alt_max, convert_alt)
            
            ###########################################
            
            ## Get the LTOOL data
            # base, top, center_of_mass, ltool_layers_cols, ltool_wls, residual_values = \
            #     get_ltool_data(new_ltool_files, ltool_var_names, alt_max, convert_alt)
            
            ## Dataframe for the LTOOL data
            # ltool_data_df = get_ltool_df(base, top, center_of_mass,
            #                               ltool_var_names, ltool_wls, ltool_layers_cols, ltool_cols)
            
            ltool_data_df = pd.DataFrame()
            for var in range(len(ltool_var_names)):
                
                if var == 0:
                    ltool_data_df = pd.DataFrame(globals()[ltool_var_names[var]], index = ltool_wls,
                                      columns = ltool_layers_cols)
                    ltool_data_df.columns = pd.MultiIndex.from_product([[ltool_cols[var]],
                                                                        ltool_data_df.columns])
        
                else:
                    ltool_data_df = ltool_data_df.join(pd.DataFrame(globals()[ltool_var_names[var]],
                                      columns = pd.MultiIndex.from_product([[ltool_cols[var]],
                                                                            ltool_layers_cols]),
                                              index = ltool_data_df.index))
            
            ##### Keep the residual layer's values in a dictionary #####
            res_layer_df = pd.DataFrame()
            res_layer_df = pd.DataFrame(residual_values, index = ltool_wls,
                                        columns = ltool_layers_cols)
            
            ## Save the LTOOL and residual layer dataframes in a dictionary
            ltool_dict["{}".format(dict_key)] = ltool_data_df
            res_layer_dict["{}".format(dict_key)] = res_layer_df
    
    
            ## --------------- 2) Get the OPTICAL data and make the plots --------------- ##
            
            ## Check whether there are bsc files, or ext files, or both
            optical_folders_temp = os.path.join(optical_dir_main,
                                                station+'/'+yr+'/'+date_path+'*/*/*/')
            
            # for a specific date folder #
            # optical_folders_temp = os.path.join(optical_dir_main,
            #                                     station+'/'+yr+'/'+date_path+f'{st_id}/*/*/')
            
            optical_folders = glob.glob(optical_folders_temp); optical_folders.sort()
            check_optical_folders = len(optical_folders)
            
            if check_optical_folders == 1:
                
                optical_folder = optical_folders[0]
                opt_files_temp = os.path.join(optical_folder, '*'+unique_ltool_filename_dates[i]+'*')
                opt_files = glob.glob(opt_files_temp); opt_files.sort()
                
                if 'bsc' in optical_folder:
                    
                    len_bsc = len(opt_files)
                    len_ext = 0
                
                else:
                    
                    len_ext = len(opt_files)
                    len_bsc = 0

            else:
            
                for optical_folder in optical_folders:
        
                    opt_files_temp = os.path.join(optical_folder,
                                                  '*'+unique_ltool_filename_dates[i]+'*')
                    opt_files = glob.glob(opt_files_temp); opt_files.sort()
                
                    if 'bsc' in optical_folder:
                        len_bsc = len(opt_files)
                    
                    else:
                        len_ext = len(opt_files)
            
            
            ## ----- a) Get the bsc data and make the plots ----- ##
            
            if len_bsc == 0:
                
                print('There are no backscatter files for', station, ', on',
                      unique_ltool_start_dates_dt[i], 'and a =', dilation)
            
            else:
                
                opt_folder_name = 'bsc'
                variable_name = 'backscatter'
                optical_var_names = ['backscatter_coef', 'altitude']
                optical_cols = ['Backscatter', 'Altitude']
                opt_col = optical_cols[0]
                units = ' [m$^{-1}$$\cdot$sr$^{-1}$]' # Units for the x axis parameter
                
                backscatter_coef, altitude, opt_wls = \
                    get_bsc_or_ext_data(alt_max, unique_ltool_filename_dates[i], optical_dir_main,
                                 station, yr, date_path, opt_folder_name, variable_name)
                
                # optical_data_df = get_bsc_or_ext_df(backscatter_coef, altitude,
                #                           optical_var_names, opt_wls, optical_cols)
                
                ## Final DataFrame for bsc data
                optical_data_df = pd.DataFrame()
                for var in range(len(optical_var_names)):
                    if var == 0:
                        optical_data_df = pd.DataFrame(globals()[optical_var_names[var]],
                                                columns = opt_wls)
                        optical_data_df.columns = pd.MultiIndex.from_product([[optical_cols[var]],
                                                                      optical_data_df.columns])
                    else:
                        optical_data_df = optical_data_df.join(pd.DataFrame(globals()[optical_var_names[var]],
                                              columns = pd.MultiIndex.from_product([[optical_cols[var]],
                                                                                    opt_wls])))
                
                ## Save the OPTICAL dataframes in a dictionary
                optical_dict["{}".format(dict_key)] = optical_data_df
                
                ## Get the common wavelengths between LTOOL and bsc files
                opt_ltl_wls = np.intersect1d(opt_wls, ltool_wls)
                check_common_wls = len(opt_ltl_wls)
                
                ## Move on and make the plots, if there are common wavelengths
                if check_common_wls == 0:
                    
                    print('There are no common wavelengths between the LTOOL and the '+
                          variable_name+' files for', station, ', on',
                          unique_ltool_filename_dates[i], 'and a =', dilation)
                
                else:
                    
                    if check_common_wls > 1:
                        
                        ## Sort the common wavelengths
                        opt_ltl_wls = sort_str_wls(opt_ltl_wls)
                    
                    ## Define the number of the plots
                    fig_cols = check_common_wls
                    
                    fig_title = station+dict_title_sep+time_duration+dict_title_sep+'a = '+dilation
                    date_text = unique_ltool_filename_dates[i]
                    
                    ## Make figure using the 'axs[ax]' commands
                    if fig_cols > 1:
                        
                        make_figure(fig_cols, fig_title, opt_ltl_wls, optical_data_df, opt_col,
                                    min_xval, max_xval, ltool_cols, units, alt_min, alt_max,
                                    ltool_data_df, convert_alt, res_layer_df)
                    
                    ## Make a single plot using the 'plt' commands
                    else:
                        
                        make_single_plot(fig_title, opt_ltl_wls, optical_data_df, opt_col,
                                    min_xval, max_xval, ltool_cols, units, alt_min, alt_max,
                                    convert_alt, ltool_data_df, res_layer_df)
                    
                    fig_nm = '_'.join(['Profile', opt_col, station, date_text, f'a={dilation}'])
                    # fig_out = path_out3+fig_nm
                    
                    # savefig(fig_out, bbox_inches = 'tight', dpi = dpi_val)
                    plt.show()
                    plt.close()
                    
            
            ## ----- b) Get the ext data and make the plots ----- ##
            
            if len_ext == 0:
                
                print('There are no extinction files for', station, ', on',
                      unique_ltool_start_dates_dt[i], 'and a =', dilation)
            
            else:
                
                opt_folder_name = 'ext'
                variable_name = 'extinction'
                optical_var_names = ['extinction_coef', 'altitude']
                optical_cols = ['Extinction', 'Altitude']
                opt_col = optical_cols[0]
                units = ' [m$^{-1}$]'
                
                extinction_coef, altitude, opt_wls = \
                    get_bsc_or_ext_data(alt_max, unique_ltool_filename_dates[i], optical_dir_main,
                                 station, yr, date_path, opt_folder_name, variable_name)
                
                # optical_data_df = get_bsc_or_ext_df(extinction_coef, altitude,
                #                           optical_var_names, opt_wls, optical_cols)
                
                ## Final DataFrame for ext data
                optical_data_df = pd.DataFrame()
                for var in range(len(optical_var_names)):
                    if var == 0:
                        optical_data_df = pd.DataFrame(globals()[optical_var_names[var]],
                                                columns = opt_wls)
                        optical_data_df.columns = pd.MultiIndex.from_product([[optical_cols[var]],
                                                                      optical_data_df.columns])
                    else:
                        optical_data_df = optical_data_df.join(pd.DataFrame(globals()[optical_var_names[var]],
                                              columns = pd.MultiIndex.from_product([[optical_cols[var]],
                                                                                    opt_wls])))
                
                ## Save the OPTICAL dataframes in a dictionary
                optical_dict["{}".format(dict_key)] = optical_data_df
                
                ## Get the common wavelengths between LTOOL and bsc files
                opt_ltl_wls = np.intersect1d(opt_wls, ltool_wls)
                check_common_wls = len(opt_ltl_wls)
                
                ## Move on and make the plots, if there are common wavelengths
                if check_common_wls == 0:
                    
                    print('There are no common wavelengths between the LTOOL and the '+
                          variable_name+' files for', station, ', on',
                          unique_ltool_filename_dates[i], 'and a =', dilation)
                
                else:
                    
                    if check_common_wls > 1:
                        
                        ## Sort the common wavelengths
                        opt_ltl_wls = sort_str_wls(opt_ltl_wls)
                    
                    ## Define the number of the plots
                    fig_cols = check_common_wls
                    
                    fig_title = station+dict_title_sep+time_duration+dict_title_sep+'a = '+dilation
                    date_text = unique_ltool_filename_dates[i]
                    
                    ## Make figure using the 'axs[ax]' commands
                    if fig_cols > 1:
                        
                        make_figure(fig_cols, fig_title, opt_ltl_wls, optical_data_df, opt_col,
                                    min_xval, max_xval, ltool_cols, units, alt_min, alt_max,
                                    ltool_data_df, convert_alt, res_layer_df)
                    
                    ## Make a single plot using the 'plt' commands
                    else:
                        
                        make_single_plot(fig_title, opt_ltl_wls, optical_data_df, opt_col,
                                    min_xval, max_xval, ltool_cols, units, alt_min, alt_max,
                                    convert_alt, ltool_data_df, res_layer_df)
                    
                    fig_nm = '_'.join(['Profile', opt_col, station, date_text, f'a={dilation}'])
                    # fig_out = path_out3+fig_nm
                    
                    # savefig(fig_out, bbox_inches = 'tight', dpi = dpi_val)
                    plt.show()
                    plt.close()
    
        # if sys_exit_parameter in folder:
        #     sys.exit()
