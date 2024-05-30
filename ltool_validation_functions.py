# -*- coding: utf-8 -*-
"""
Created on Sat May 28 09:29:16 2022

@author: lenovo-pc
"""

import os
import glob
import numpy
import sys
import shutil
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



'''
---------------------------------------------
1) Functions for ltool_boxplots_hists script
---------------------------------------------
'''

def replace_inf(x):
    
    x[np.logical_or(x == np.inf, x == -np.inf)] = np.nan
    
    return x


def filters_bsc_ext(lidar, alt, k, x, alt_max, xmin, xmax): ## For profiles
    
    ## Initial
    bsc_ext = lidar[k, x, :][np.logical_and(lidar[k, x, :] >= xmin, lidar[k, x, :] <= xmax)]
    height = alt[np.where([np.logical_and(lidar[k, x, :] >= xmin, lidar[k, x, :] <= xmax)])]
    
    return bsc_ext, height


def filters_ae_bae(x, alt, k, alt_max, xmin, xmax): ## For profiles
    
    ## Initial
    y = x[k, :][np.logical_and(x[k, :] >= xmin, x[k, :] <= xmax)] # BAE or AE
    h = alt[np.where(np.logical_and(x[k, :] >= xmin, x[k, :] <= xmax))] # Altitude
    
    return y, h


def filters_lr(lr, alt, k, alt_max, xmin, xmax): ## For profiles
    
    ## Initial
    l = lr[k, :][np.logical_and(lr[k, :] >= xmin, lr[k, :] <= xmax)] # LR
    al = alt[np.where(np.logical_and(lr[k, :] >= xmin, lr[k, :] <= xmax))] # Altitude
    
    return l, al


def filters_df(x, minval, maxval): ## For the 'all_std_df' variable
    
    x[np.logical_and(x < minval, x > maxval)] = np.nan    

    return x



'''
---------------------------------------------------------------
2) Functions for ltool_validation_profiles (loop/noloop) script
---------------------------------------------------------------
'''

def make_figure(fig_cols, fig_title, opt_ltl_wls, optical_data_df, opt_col, min_xval, max_xval,
                ltool_cols, units, alt_min, alt_max, ltool_data_df, convert_alt, res_layer_df):
    
    #### 1) Define the plot's parameters ####
    
    fig_size = (13, 8)
    font_size_label = 12
    font_size_title = 15
    font_size_annotate = 10
    font_weight_label = 'bold'
    anchor_pos = (0.5, 1)
    ylabel_text = 'Altitude [m]'
    alpha = 0.3
    axspan_color = 'grey'
    
    
    #### 2) Make the plot ####
    
    fig, axs = plt.subplots(1, fig_cols, figsize = fig_size) # Define the figure
                    
    # fig, axs = plt.subplots(1, 2, figsize = fig_size) ## TEST!!
    fig.suptitle(fig_title, fontsize = font_size_title,
                 fontweight = font_weight_label) # Title
            
    ax = 0 # ltool_wls[ax]
    for opt_ltl_wl in opt_ltl_wls: # Make the sublots (ax in range(len(axs)))
                            
        y_val_temp = optical_data_df['Altitude'][opt_ltl_wl]
        x_val_temp = optical_data_df[opt_col][opt_ltl_wl]
                
        y_val_temp2 = y_val_temp[y_val_temp <= alt_max]
        x_val_temp2 = x_val_temp.values[np.where(y_val_temp <= alt_max)]
                            
        ## Define the color of the line in the profiles
        line_color = get_plot_color(opt_ltl_wl)
        
        ## a) Plot the bsc or ext profiles
        axs[ax].plot(x_val_temp2, y_val_temp2, label = opt_ltl_wl,
                     color = line_color)
                
        axs[ax].set_xlabel(opt_col+' coef\n'+units, fontsize = font_size_label,
                           fontweight = font_weight_label) # 'Aerosol '+opt_col+' Coefficient'
        axs[ax].set_ylabel(ylabel_text, fontsize = font_size_label,
                           fontweight = font_weight_label)
                        
        axs[ax].set_xlim(min_xval, max_xval)
        axs[ax].set_ylim(alt_min, alt_max)
        axs[ax].legend(bbox_to_anchor = anchor_pos, loc = 'lower center',
                       prop = FontProperties())
                          
        ## b) Plot the layers
        for ltool_col in ltool_cols:
                             
            # layer_line_color, layer_line_style, layer_text = \
            #     get_layer_plot_format(ltool_col)
                    
            temp_len = len(ltool_data_df[ltool_col].loc[opt_ltl_wl].dropna())
            annotate_text = max_xval # For command set_xlim() (2)
                                
            for l in range(temp_len): # ltool_data_df[ltool_col].columns
                                
                layer_temp = ltool_data_df[ltool_col].columns[l]
                # layer_pbl_text = layer_temp
                                
                layer_val = ltool_data_df[ltool_col][layer_temp].loc[opt_ltl_wl]*\
                    convert_alt
                
                pbl = res_layer_df[layer_temp].loc[opt_ltl_wl]
                
                ##### axspan command instead of hlines command #####
                                
                layer_top = ltool_data_df['Top'][layer_temp].loc[opt_ltl_wl]*\
                    convert_alt
                layer_b = ltool_data_df['Base'][layer_temp].loc[opt_ltl_wl]*\
                    convert_alt
                
                if pbl == 1:
                    
                    pbl_line_color = 'black'
                    pbl_line_style = '--'
                    
                    axs[ax].hlines(y = layer_top, xmin = min_xval,
                                   xmax = max_xval, color = pbl_line_color,
                                   linestyle = pbl_line_style)
                
                else:
                    
                    axs[ax].axhspan(layer_b, layer_top, alpha = alpha,
                                color = axspan_color)
                
                # axs[ax].axhspan(layer_b, layer_top, alpha = alpha,
                #                 color = axspan_color)
                                
                #################  New approach  #################
                                    
                if ltool_col == 'Center of mass':
                    
                    if pbl != 1:
                        layer_line_color = 'orange'
                        layer_line_style = '-'
                    
                        axs[ax].hlines(y = layer_val, xmin = min_xval,
                                       xmax = max_xval, color = layer_line_color,
                                       linestyle = layer_line_style)
                    
                    
                    ### Check if the layer is the PBL ###
                                
                    if res_layer_df[layer_temp].loc[opt_ltl_wl] == 1:
                                        
                        axs[ax].annotate('PBL',
                                         xy = (annotate_text, layer_top),
                                         xytext = (annotate_text, layer_top),
                                         fontsize = font_size_annotate,
                                         fontweight = font_weight_label)
                                    
                    else:
                            
                        if res_layer_df['L1'].loc[opt_ltl_wl] == 1:
                                            
                            layer_temp = ltool_data_df[ltool_col].columns[l-1]
                                            
                            axs[ax].annotate(layer_temp,
                                             xy = (annotate_text, layer_val),
                                             xytext = (annotate_text, layer_val),
                                             fontsize = font_size_annotate,
                                             fontweight = font_weight_label)
                        
                        else:
                            
                            axs[ax].annotate(layer_temp,
                                             xy = (annotate_text, layer_val),
                                             xytext = (annotate_text, layer_val),
                                             fontsize = font_size_annotate,
                                             fontweight = font_weight_label)
                                
                ############################################################
    
        ax += 1
                
    if fig_cols > 1:
        for ax2 in axs.flat:
            ax2.label_outer()


def make_single_plot(fig_title, opt_ltl_wls, optical_data_df, opt_col, min_xval, max_xval,
                     ltool_cols, units, alt_min, alt_max, convert_alt, ltool_data_df, res_layer_df):
    
    #### 1) Define the plot's parameters ####
    font_size_annotate = 10
    font_weight_label = 'bold'
    ylabel_text = 'Altitude [m]'
    alpha = 0.3
    axspan_color = 'grey'
    
    #### 2) Make the plot ####
    
    opt_ltl_wl = opt_ltl_wls[0]
                        
    y_val_temp = optical_data_df['Altitude'][opt_ltl_wl]
    x_val_temp = optical_data_df[opt_col][opt_ltl_wl]
    
    y_val_temp2 = y_val_temp[y_val_temp <= alt_max]
    x_val_temp2 = x_val_temp.values[np.where(y_val_temp <= alt_max)]
                        
    ## Define the color of the line in the profiles
    line_color = get_plot_color(opt_ltl_wl)
    
    ## a) Plot the bsc or ext profiles
    plt.plot(x_val_temp2, y_val_temp2, label = opt_ltl_wl,
             color = line_color)
                
    plt.xlabel(opt_col+' coef'+units,
               fontweight = font_weight_label)
    plt.ylabel(ylabel_text,
               fontweight = font_weight_label)
                        
    plt.title(fig_title, fontweight = font_weight_label)
                        
    plt.xlim(min_xval, max_xval)
    plt.ylim(alt_min, alt_max)
    plt.legend(loc = 'best')
    
    ## b) Plot the layers                   
    for ltool_col in ltool_cols:
                            
        # layer_line_color, layer_line_style, layer_text = \
        #     get_layer_plot_format(ltool_col)
                            
        temp_len = len(ltool_data_df[ltool_col].loc[opt_ltl_wl].dropna())
        annotate_text = max_xval # For command set_xlim() (2)
                            
        for l in range(temp_len): # ltool_data_df[ltool_col].columns
                                
            layer_temp = ltool_data_df[ltool_col].columns[l]
            # layer_pbl_text = layer_temp
                                
            layer_val = ltool_data_df[ltool_col][layer_temp].loc[opt_ltl_wl]*\
                convert_alt
            
            pbl = res_layer_df[layer_temp].loc[opt_ltl_wl]
                               
            ##### axspan command instead of hlines command #####
                                
            layer_top = ltool_data_df['Top'][layer_temp].loc[opt_ltl_wl]*\
                convert_alt
            layer_b = ltool_data_df['Base'][layer_temp].loc[opt_ltl_wl]*\
                convert_alt
                                
            # plt.axhspan(layer_b, layer_top, alpha = alpha,
            #             color = axspan_color)
                                
            ####################################################
            
            if pbl == 1:
                    
                pbl_line_color = 'black'
                pbl_line_style = '--'
                    
                plt.hlines(y = layer_top, xmin = min_xval,
                               xmax = max_xval, color = pbl_line_color,
                               linestyle = pbl_line_style)
                
            else:
                    
                plt.axhspan(layer_b, layer_top, alpha = alpha,
                                color = axspan_color)
                   
            if ltool_col == 'Center of mass':
                
                if pbl != 1:
                
                    layer_line_color = 'orange'
                    layer_line_style = '-'
                  
                    plt.hlines(y = layer_val, xmin = min_xval,
                               xmax = max_xval, color = layer_line_color,
                               linestyle = layer_line_style)
                
                   
                ### Check if the layer is PBL ###
                              
                if res_layer_df[layer_temp].loc[opt_ltl_wl] == 1:
                                        
                    plt.annotate('PBL',
                                 xy = (annotate_text, layer_top),
                                 xytext = (annotate_text, layer_top),
                                 fontsize = font_size_annotate,
                                 fontweight = font_weight_label)
                                    
                else:
                    
                    if res_layer_df['L1'].loc[opt_ltl_wl] == 1:
                        
                        layer_temp = ltool_data_df[ltool_col].columns[l-1]
                                            
                        plt.annotate(layer_temp,
                                     xy = (annotate_text, layer_val),
                                     xytext = (annotate_text, layer_val),
                                     fontsize = font_size_annotate,
                                     fontweight = font_weight_label)
                    
                    else:
                        plt.annotate(layer_temp,
                                     xy = (annotate_text, layer_val),
                                     xytext = (annotate_text, layer_val),
                                     fontsize = font_size_annotate,
                                     fontweight = font_weight_label)


# def get_layer_plot_format(ltl_col):
    
#     if ltl_col == 'Base':
        
#         layer_color = 'black'
#         layer_style = 'dotted'
#         layer_txt = 'B'
    
#     elif ltl_col == 'Top':
        
#         layer_color = 'black'
#         layer_style = 'dotted'
#         layer_txt = 'T'
    
#     else:
        
#         layer_color = 'orange'
#         layer_style = '-'
#         layer_txt = 'CM'
    
#     return layer_color, layer_style, layer_txt



'''
---------------------------------------------------------------
3) Functions for ltool_validation_scatters (loop/noloop) script
---------------------------------------------------------------
'''

def get_common_com_wls(first_dkey, com_dict, com_dkeys):
    
    if len(com_dict) == 1:
        com_df_temp = com_dict[com_dkeys[0]]
        common_com_df_wls = com_df_temp.index
    else:
        first_com_df = com_dict[first_dkey]
        common_com_df_wls = first_com_df.index
        ## min_com_wls = len(first_com_df)
    
        ## Get the minimum (common) number of wavelengths
        for key in com_dkeys[1:]:
            com_df_temp = com_dict[key]
            ## com_wls_temp = len(com_df_temp)
        
            com_wls_temp = com_df_temp.index
        
            common_com_df_wls = np.intersect1d(common_com_df_wls, com_wls_temp)
    
    if len(common_com_df_wls) > 1:
        
        ## Sort the wavelengths
        common_com_df_wls = sort_str_wls(common_com_df_wls)
    
    return common_com_df_wls


def get_layers_number(dil_values, common_wls, d_keys, c_dict, res_layer_dict):
    
    layers_num_temp_df = pd.DataFrame(index = dil_values)
    for common_wl in common_wls: ## [1:] --> DO NOT INCLUDE 355 nm (for ILRC 2022)
        layers_num = []
        for k in range(len(d_keys)):
            key = d_keys[k]
            
            temp_com_df = c_dict[key]
            temp_res_layer_df = res_layer_dict[key]
            
            if 1 in temp_res_layer_df.loc[common_wl].values:
                layers_num.append(len(temp_com_df['Base'].loc[common_wl].dropna()) - 1)
            
            else:
                layers_num.append(len(temp_com_df['Base'].loc[common_wl].dropna()))
                
        layers_num_temp_df[common_wl] = layers_num
            
    layers_num_temp_df['Std'] = layers_num_temp_df.std(axis = 1) # Std for each a
    
    return layers_num_temp_df


def match_layers(rs, cols, c_dict, dkeys, com_wls, dil_vals_int):
    
    first_layer = 'L1'
           
    if cols == 1:
        first_wl = com_wls[0]
        match_layers_vals = np.nan*np.zeros((rs))
                
        for r in range(rs):
                
            key = dkeys[r]
            temp_com_df = c_dict[key]
            match_layers_vals[r] = temp_com_df.loc[first_wl, first_layer]
                
    else:
        match_layers_vals = np.nan*np.zeros((rs, cols))
        
        for r in range(rs):
            
            key = dkeys[r]
            temp_com_df = c_dict[key]
            
            for c in range(cols):
                
                com_wl = com_wls[c]
                
                if c == 0:
                    ## Get the 1st layer for the first common wl, for each a
                    first_val = temp_com_df.loc[com_wl, first_layer]
                    match_layers_vals[r, c] = first_val
                    previous_val = first_val
                else:
                    layer_temp_vals = temp_com_df.loc[com_wl].values
                    
                    ## Find the position of the desired layer
                    layer_diff = abs(previous_val - layer_temp_vals)
                    layer_pos = np.where(layer_diff == min(layer_diff))[0]
                    
                    ## Get the value of the desired layer
                    new_layer_val = temp_com_df.loc[com_wl, 'L'+str(layer_pos[0]+1)]
                    match_layers_vals[r, c] = new_layer_val
                    previous_val = first_val = new_layer_val
            
    match_lrs_df = pd.DataFrame(match_layers_vals, index = dil_vals_int, columns = com_wls)
    
    return match_lrs_df



'''
----------------------------------------------------------------
4) Functions for ltool_validation_temporal_plots (noloop) script
----------------------------------------------------------------
'''

def make_temporal_plot(ltool_layers_cols, df, dot_color, fb_color, alpha, convert_alt, ax):
    
    for layer in ltool_layers_cols: # [1:]
    
        base_vals = df['Base'][layer]*convert_alt
        top_vals = df['Top'][layer]*convert_alt
        com_vals = df['Center of mass'][layer]*convert_alt
    
        ax.scatter(df.index, com_vals, color = dot_color)
        ax.fill_between(df.index, base_vals, top_vals, color = fb_color, alpha = alpha)


def split_first_layer(dates_dt, res_vals_df, l1, df, convert_alt):
    
    tp_of_pbl = []; dts_dt_pbl = []
    bs_l1_layer = []; tp_l1_layer = []; comss_l1_layer = []; dts_dt_l1_layer = []
    
    for date_dt in dates_dt:
            
        pbl_val = res_vals_df[l1].loc[date_dt]
            
        if pbl_val == 1:
                
            tp_of_pbl_val = df['Top'][l1].loc[date_dt]*convert_alt
                
            tp_of_pbl.append(tp_of_pbl_val)
            dts_dt_pbl.append(date_dt)
            
        else:
                
            bs_l1_layer.append(df['Base'][l1].loc[date_dt])
            tp_l1_layer.append(df['Top'][l1].loc[date_dt])
            comss_l1_layer.append(df['Center of mass'][l1].loc[date_dt])
            dts_dt_l1_layer.append(date_dt)
    
    return tp_of_pbl, dts_dt_pbl, bs_l1_layer, tp_l1_layer, comss_l1_layer, dts_dt_l1_layer



'''
---------------------------------------------------------------------------------------
5) Functions for ltool_validation_profiles, ltool_validation_scatters (loop/noloop) and
   ltool_validation_temporal_plots (noloop) scripts
---------------------------------------------------------------------------------------
'''

def get_rows_columns(new_ltl_files):
    
    ## Get the number of the layers and the wavelengths from the LTOOL files ##
    ltool_layers = []
    ltl_wls = [] # ltool_wls
    for new_ltool_file in new_ltl_files:
        fh = Dataset(new_ltool_file, mode = 'r')
            
        ltool_layers.append(len(fh.variables['base']))
            
        ltool_wl_temp = os.path.splitext(new_ltool_file)[0].split('\\')[-1].split('_')[2]
        ltool_wl = get_wls(ltool_wl_temp)
        ltl_wls.append(ltool_wl)
    
    ltool_layers_final = max(ltool_layers) # min(ltool_layers)
    ## ltool_lrs_cols = ['L'+str(l) for l in range(1, ltool_layers_final + 1)]
    ltool_wls_len = len(ltl_wls)
    
    rs = ltool_wls_len
    clmns = ltool_layers_final
    
    return rs, clmns, ltl_wls


def get_ltool_data(new_ltl_files, rows, columns, ltl_var_names, altitude_max, conv_alt): # NEW!!
    
    #### a) Get the data from the LTOOL files - Get the number of the layers automatically (ignore b)

    ## Wavelenghts x Layers
    bs = np.nan*np.zeros((rows, columns))
    tp = np.nan*np.zeros((rows, columns))
    com = np.nan*np.zeros((rows, columns))
    res_vals = np.nan*np.zeros((rows, columns))
    
    for k in range(rows): # ltool_wls_len
        fh = Dataset(new_ltl_files[k], mode = 'r')
        for m in range(columns): # ltool_layers_final
            
            if m < len(fh.variables['residual_layer_flag']):
                res_vals[k, m] = fh.variables['residual_layer_flag'][:][m]
                
            for var in ltl_var_names:
                if m < len(fh.variables[var]):
                    if var == 'base':
                        bs[k, m] = fh.variables[var][:][m]
                    elif var == 'top':
                        tp[k, m] = fh.variables[var][:][m]
                    else:
                        com[k, m] = fh.variables[var][:][m]
    
    ## Filter for top <= 5000 m
    bs_final = np.nan*np.zeros((rows, columns))
    tp_final = np.nan*np.zeros((rows, columns))
    com_final = np.nan*np.zeros((rows, columns))
    res_vals_final = np.nan*np.zeros((rows, columns))
    
    for r in range(rows):
        for c in range(columns):
            if tp[r, c] <= altitude_max/conv_alt:
                bs_final[r, c] = bs[r, c]
                tp_final[r, c] = tp[r, c]
                com_final[r, c] = com[r, c]
                res_vals_final[r, c] = res_vals[r, c]
    
    ## Make the layers and the final variables
    ltool_lrs_cols_temp = []
    for r in range(rows):
        len_temp = len(bs_final[r][~numpy.isnan(bs_final[r])])
        ## layer_temp = ['L'+str(l) for l in range(1, len_temp + 1)]
    
        ltool_lrs_cols_temp.append(len_temp)
    
    ltool_layers_final_new = max(ltool_lrs_cols_temp)
    columns_new = ltool_layers_final_new
    ltool_lrs_cols = ['L'+str(l) for l in range(1, columns_new + 1)]
    
    ## Wavelengths x Max layers up to 5000 m
    bs_final2 = np.nan*np.zeros((rows, columns_new))
    tp_final2 = np.nan*np.zeros((rows, columns_new))
    com_final2 = np.nan*np.zeros((rows, columns_new))
    res_vals_final2 = np.nan*np.zeros((rows, columns_new))
    
    for r in range(rows):
        for lrs in range(columns_new):
            if lrs < len(bs_final[r][~numpy.isnan(bs_final[r])]):
                bs_final2[r, lrs] = bs_final[r, lrs]
                tp_final2[r, lrs] = tp_final[r, lrs]
                com_final2[r, lrs] = com_final[r, lrs]
                res_vals_final2[r, lrs] = res_vals_final[r, lrs]
    
    
    #### b) Get the data from the LTOOL files - Insert the number of layers by hand (ignore a)
    '''
    ltool_layers_final = 3
    ltool_lrs_cols = ['L'+str(l) for l in range(1, ltool_layers_final + 1)]
    ltool_wls_len = len(ltl_wls)
    
    rows = ltool_wls_len
    columns = ltool_layers_final
    
    bs_final2 = np.nan*np.zeros((rows, columns))
    tp_final2 = np.nan*np.zeros((rows, columns))
    com_final2 = np.nan*np.zeros((rows, columns))
    
    for k in range(ltool_wls_len):
        fh = Dataset(new_ltool_files[k], mode = 'r')
        for m in range(ltool_layers_final):
            bs_final2[k, m] = fh.variables['base'][:][m]
            tp_final2[k, m] = fh.variables['top'][:][m]
            com_final2[k, m] = fh.variables['center_of_mass'][:][m]
    '''
    
    return bs_final2, tp_final2, com_final2, ltool_lrs_cols, res_vals_final2


def get_ltool_df(base, top, center_of_mass, var_nms, wls, cols, main_cols):
    
    #### Final DataFrame for LTOOL data ####
    df = pd.DataFrame()
    
    for var in range(len(var_nms)):
        
        if var == 0:
            
            df = pd.DataFrame(globals()[var_nms[var]], index = wls, columns = cols)
            df.columns = pd.MultiIndex.from_product([[main_cols[var]], df.columns])
        
        else:
            
            df = df.join(pd.DataFrame(globals()[var_nms[var]],
                                      columns = pd.MultiIndex.from_product([[main_cols[var]], cols]),
                                              index = df.index))
    
    return df


def get_optical_data(opt_files, al_max):
            
    ## 1) Get each file's length and the wavelength
    optical_file_len = []
    optical_wls = []
    for optical_file in opt_files:
        
        fh = Dataset(optical_file, mode = 'r')
        
        ## Keep the length of the file where the altitude is lower than 5000 m
        alt_temp = fh.variables['altitude'][:]
        alt_temp_filterd = alt_temp[alt_temp <= al_max]
        optical_file_len.append(len(alt_temp_filterd))
        ## optical_file_len.append(len(fh.variables['altitude']))
        
        ## Keep the wavelength for each file
        optical_wl_temp = os.path.splitext(optical_file)[0].split('\\')[-1].split('_')[2]
        optical_wl = get_wls(optical_wl_temp)
        optical_wls.append(optical_wl)
     
    optical_file_len_max = max(optical_file_len) # Get the maximum length of the values
    ## from the OPTICAL files
    optical_wls_len = len(optical_wls) # The number of the wavelengths (355, 532 and 1064)
    
    ## 2) Get the data from the OPTICALfiles
    ## make_vars = np.nan*np.zeros((optical_file_len_min, optical_wls_len)) # Min number values x Wavelenghts
    
    rows = optical_file_len_max
    columns = optical_wls_len
    
    ## Min number values x Wavelenghts
    bsc_coef = np.nan*np.zeros((rows, columns))
    ext_coef = np.nan*np.zeros((rows, columns))
    alt = np.nan*np.zeros((rows, columns))
    
    for n in range(optical_wls_len):
        
        fh = Dataset(opt_files[n], mode = 'r')
        temp_fh_len = optical_file_len[n]
        
        for j in range(temp_fh_len):
            alt[j, n] = fh.variables['altitude'][:][j] # In m
            
        if 'backscatter' in fh.variables:
            for j in range(temp_fh_len):
                bsc_coef[j, n] = fh.variables['backscatter'][0][0][j]
            
        if 'extinction' in fh.variables:
            for j in range(temp_fh_len):
                ext_coef[j, n] = fh.variables['extinction'][0][0][j]
    
    return bsc_coef, ext_coef, alt, optical_wls


def get_bsc_or_ext_data(al_max, unique_date, optical_dir, station, yr, date_path, opt_folder_name,
                        variable_name):
    
    opt_files_temp = os.path.join(optical_dir, station+'/'+yr+'/'+date_path+'*/*/'+\
                                  opt_folder_name+'/*'+unique_date+'*')
    opt_files = glob.glob(opt_files_temp)
    
    
    ## ------- 1) Get each file's length and wavelength ------- ##
    
    optical_file_len = []
    optical_wls = []
    
    for optical_file in opt_files:
        
        fh = Dataset(optical_file, mode = 'r')
        
        ## Keep the length of the file where the altitude is lower than 5000 m
        alt_temp = fh.variables['altitude'][:] #.data
        alt_temp_filterd = alt_temp[alt_temp <= al_max]
        optical_file_len.append(len(alt_temp_filterd))
        ## optical_file_len.append(len(fh.variables['altitude']))
        
        ## Keep the wavelength for each file
        optical_wl_temp = os.path.splitext(optical_file)[0].split('\\')[-1].split('_')[2]
        optical_wl = get_wls(optical_wl_temp)
        optical_wls.append(optical_wl)
    
    ## Get the maximum length of the values from the OPTICAL files
    optical_file_len_max = max(optical_file_len)
    
    ## The number of the wavelengths (355, 532 and 1064)
    optical_wls_len = len(optical_wls)
    
    
    ## ------- 2) Get the data from the OPTICAL files ------- ##
    
    # Min number values x Wavelenghts
    ## make_vars = np.nan*np.zeros((optical_file_len_min, optical_wls_len))
    
    rows = optical_file_len_max
    columns = optical_wls_len
    
    ## Min number values x Wavelenghts
    optical_propertie = np.nan*np.zeros((rows, columns)) # Bsc or Ext
    alt = np.nan*np.zeros((rows, columns))
    
    for n in range(optical_wls_len):
        
        fh = Dataset(opt_files[n], mode = 'r')
        temp_fh_len = optical_file_len[n]
        
        for j in range(temp_fh_len):
            
            optical_propertie[j, n] = fh.variables[variable_name][0][0][j]
            alt[j, n] = fh.variables['altitude'][:][j] # In m
    
    return optical_propertie, alt, optical_wls


def get_bsc_or_ext_df(propertie, altitude, var_nms, wls, main_cols):
    
    #### Final DataFrame for OPTICAL data ####
    df = pd.DataFrame()
    
    for var in range(len(var_nms)):
        
        if var == 0:
            
            df = pd.DataFrame(globals()[var_nms[var]], columns = wls)
            df.columns = pd.MultiIndex.from_product([[main_cols[var]], df.columns])
        
        else:
            
            df = df.join(pd.DataFrame(globals()[var_nms[var]],
                                      columns = pd.MultiIndex.from_product([[main_cols[var]],
                                                                                    wls])))
    
    return df


def get_lr_bae_ae(opt_df, opt_cols):
    ## make_vars = np.nan*np.zeros((len(opt_df), len(opt_df['Backscatter'].columns))) # opt_df.shape
    
    columns_names = opt_df['Backscatter'].columns
    
    rows = len(opt_df)
    columns = len(columns_names)
    
    ## Min number values (same with the above function) x Wavelengths
    lidar_ratio = np.nan*np.zeros((rows, columns))
    bsc_ae = np.nan*np.zeros((rows, columns))
    angs_exp = np.nan*np.zeros((rows, columns))
    ## alt = np.nan*np.zeros((rows, columns))
    
    ## Calculate the LR
    for r in range(rows):
        for c in range(columns):
            ext_temp = opt_df['Extinction'][columns_names[c]][r]
            bsc_temp = opt_df['Backscatter'][columns_names[c]][r]
            
            lidar_ratio[r, c] = ext_temp/bsc_temp
    
    ## Calculate the BAE and AE
    wl_sep = '/'
    
    if columns == 1:
      bsc_ae_angs_exp_wls = ['None'] # columns_names.values[0]
    elif columns == 2:
        new_bae_ae_columns = columns - 1
        bsc_ae = np.nan*np.zeros((rows, new_bae_ae_columns))
        angs_exp = np.nan*np.zeros((rows, new_bae_ae_columns))
        
        ## Make the BAE and AE wavelenghts (columns)
        bsc_ae_angs_exp_wls = []
        for wl in range(new_bae_ae_columns):
            wl1_temp = columns_names[wl].split(' ')[0]
            wl2_temp = columns_names[wl+1].split(' ')[0]
            
            bsc_ae_angs_exp_wls.append(wl1_temp+wl_sep+wl2_temp)
            
        for r in range(rows):
            for c in range(new_bae_ae_columns):
                wl1 = float(columns_names[c].split(' ')[0])
                wl2 = float(columns_names[c+1].split(' ')[0])
                
                for op_col in opt_cols[:-1]:
                    temp1 = opt_df[op_col][columns_names[c]][r]
                    temp2 = opt_df[op_col][columns_names[c+1]][r] 
                    
                    if op_col == 'Backscatter': # Calculate the BAE
                        bsc_ae[r, c] = - np.log(temp1/temp2)/np.log(wl1/wl2)
                    else: # Calculate the AE
                        angs_exp[r, c] = - np.log(temp1/temp2)/np.log(wl1/wl2)
    else:
        ## Make the BAE and AE wavelenghts (columns)
        bsc_ae_angs_exp_wls = []
        for wl in range(columns):
            if wl != columns - 1:
                wl1_temp = columns_names[wl].split(' ')[0]
                wl2_temp = columns_names[wl+1].split(' ')[0]
            
                bsc_ae_angs_exp_wls.append(wl1_temp+wl_sep+wl2_temp)
            else:
                wl1_temp = columns_names[0].split(' ')[0]
                wl2_temp = columns_names[wl].split(' ')[0]
            
                bsc_ae_angs_exp_wls.append(wl1_temp+wl_sep+wl2_temp)
        
        for r in range(rows):
            for c in range(columns):
                if c != columns - 1:
                    wl1 = float(columns_names[c].split(' ')[0])
                    wl2 = float(columns_names[c+1].split(' ')[0])
                    
                    for op_col in opt_cols[:-1]:
                        temp1 = opt_df[op_col][columns_names[c]][r]
                        temp2 = opt_df[op_col][columns_names[c+1]][r] 
                    
                        if op_col == 'Backscatter': # Calculate the BAE
                            bsc_ae[r, c] = - np.log(temp1/temp2)/np.log(wl1/wl2)
                        else: # Calculate the AE
                            angs_exp[r, c] = - np.log(temp1/temp2)/np.log(wl1/wl2)
                else:
                    wl1 = float(columns_names[0].split(' ')[0])
                    wl2 = float(columns_names[c].split(' ')[0])
                    
                    for op_col in opt_cols[:-1]:
                        temp1 = opt_df[op_col][columns_names[0]][r]
                        temp2 = opt_df[op_col][columns_names[c]][r]
                    
                        if op_col == 'Backscatter': # Calculate the BAE
                            bsc_ae[r, c] = - np.log(temp1/temp2)/np.log(wl1/wl2)
                        else: # Calculate the AE
                            angs_exp[r, c] = - np.log(temp1/temp2)/np.log(wl1/wl2)
    
    return lidar_ratio, bsc_ae, angs_exp, bsc_ae_angs_exp_wls



'''
----------------------------
6) Functions for general use
----------------------------
'''

def sort_str_wls(wls):
    
    if len(wls) == 2 and '1064 nm' in wls:
        wls[0], wls[1] = wls[1], wls[0]
        
    else: # if len(opt_ltl_wls) == 3
        for w in range(len(wls) - 1):
            wls[w], wls[w+1] = wls[w+1], wls[w]
    
    return wls


def get_wls(raw_wl):
    
    if raw_wl == '0355':
        wavelength = '355 nm'
    elif raw_wl == '0532':
        wavelength = '532 nm'
    else:
        wavelength = '1064 nm'
    
    return wavelength


def get_plot_color(wl):
    
    color_355 = 'blue'
    color_532 = 'green'
    color_1064 = 'red'
    
    if wl == '355 nm':
        plot_color = color_355
    
    elif wl == '532 nm':
        plot_color = color_532
    
    else:
        plot_color = color_1064
    
    return plot_color

def make_ltl_opt_folders(fls, ltl_opt, pars_lst):
    
    '''
    This function makes the required folder(s) format for the LTOOL and the Optical (backscatter or
    extinction) data.
    The required format for the LTOOL data is .../station/year/day-st_code/dilation_folder/files, e.g.
    .../Potenza/2021/20210402pot1931/a_200/files.
    The required format for the Optical data is .../station/year/day-st_code/day-st_code/opt_folder/files,
    e.g. .../Potenza/2019/20190307po77/20190307po77/bsc (or ext)/files
    
    INPUTS:
    fls: array/list of the LTOOL or Optical files (str)
    ltl_opt: choose between the LTOOL or the Optical folders/files, depending what you want (str, valid
    values: 'ltl' or 'opt')
    pars_lst: list which contains specific values:
    for ltl_opt = 'ltl' --> pars_lst = [split_par, dil_folder]
    for ltl_opt = 'opt' --> pars_lst = [split_par]
    where split_par is a parameter to split the folder names (str, e.g. '/') and dil_folder is the
    dilation folder (str, in foramt a_dilation, for dilation = 200, 300, etc.) and
    
    OUTPUT:
    the required folder format for either LTOOL, or Optical data
    
    NOTES:
    In order for this function to work, the LTOOL and Optical files must be in the year folder. Some
    changes might be necessary in the .split() commands!!!
    '''
    
    ## Sort the files ##
    fls.sort()
    
    ## Get the parameters ##
    split_par = pars_lst[0]
    
    ## Find the unique day-station codes ##
    day_st = []
    for f in fls:
        day_st.append(f.split(split_par)[-1].split('_')[6])
    day_st = list(set(day_st)); day_st.sort()
    
    ## Keep the main data folder ##
    folder = f.split(split_par)[0]+'/' # use the last file from the previous loop (same for any other file)
    
    if ltl_opt == 'opt':
        
        for fl_nm in day_st:
    
            ## Make the output folder(s) ##
            day_folder = os.path.join(folder, fl_nm+'/')
            if not os.path.exists(day_folder):
                os.mkdir(day_folder)
            
            day_folder2 = os.path.join(day_folder, fl_nm+'/')
            if not os.path.exists(day_folder2):
                os.mkdir(day_folder2)
            
            # day_val = fl_nm[:8] # keep only the day
            
            ## Get the specific day-station code files ##
            files = glob.glob(folder+f'*{fl_nm}*.nc'); files.sort()
            
            ## Get the optical code from the files' name ##
            file_codes = []
            for file in files:
                file_codes.append(file.split(split_par)[-1].split('_')[1])
            file_codes = list(set(file_codes)); file_codes.sort()
        
            ## Make the bsc and/or the ext folder(s) and move the files to the
            ## respective folders
            if '002' in file_codes:
                
                opt_folder = os.path.join(day_folder2, 'ext/')
                if not os.path.exists(opt_folder):
                    os.mkdir(opt_folder)
                
                opt_files = os.path.join(folder, f'*_002_*{fl_nm}*.nc')
                opt_files = glob.glob(opt_files); opt_files.sort()
                for file in opt_files:
                    shutil.move(file, opt_folder)
            
            if '000' in file_codes or '003' in file_codes:
                        
                opt_folder = os.path.join(day_folder2, 'bsc/')
                if not os.path.exists(opt_folder):
                    os.mkdir(opt_folder)
                
                if '000' in file_codes:
                    opt_files = os.path.join(folder, f'*_000_*{fl_nm}*.nc')
                    opt_files = glob.glob(opt_files); opt_files.sort()
                    for file in opt_files:
                        shutil.move(file, opt_folder)
                
                if '003' in file_codes:
                    opt_files = os.path.join(folder, f'*_003_*{fl_nm}*.nc')
                    opt_files = glob.glob(opt_files); opt_files.sort()
                    for file in opt_files:
                        shutil.move(file, opt_folder)
    
    else:
        
        dil_folder = pars_lst[1]
        
        for fl_nm in day_st:
    
            ## Make the output folder(s) ##
            day_folder = os.path.join(folder, fl_nm+'/')
            if not os.path.exists(day_folder):
                os.mkdir(day_folder)
            
            dilation_folder = os.path.join(day_folder, dil_folder+'/')
            if not os.path.exists(dilation_folder):
                os.mkdir(dilation_folder)
            
            ## Get the specific day-station code files ##
            files = glob.glob(folder+f'*{fl_nm}*.nc'); files.sort()
            
            ## Move the files to the respective folders ##
            for file in files:
                shutil.move(file, dilation_folder)
