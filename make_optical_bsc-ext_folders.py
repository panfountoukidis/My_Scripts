# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:20:39 2024

@author: lenovo-pc
"""


import os
import glob
import sys
import shutil


## Define the input folders ##
main_path = 'C:/Users/lenovo-pc/Documents/Data_In_Out/' # Test!!
optical_dir_main = os.path.join(main_path, 'ltools_v2.0/input/SCC/OPTICAL_products_New_Format/')


## Define some general parameters ##
station = 'Potenza'; yr = 2021

## Get the day folders ##
day_folders = os.path.join(optical_dir_main, f'{station}/{yr}/*/')
day_folders = glob.glob(day_folders); day_folders.sort()
day_folders = day_folders[3:] # Test!!

for folder in day_folders:
    
    ## Keep the codes from the files' names ##
    temp_files = os.path.join(folder, '*/*')
    temp_files = glob.glob(temp_files); temp_files.sort()
    
    file_codes = []
    for file in temp_files:
        file_codes.append(file.split('\\')[-1].split('_')[1])
    file_codes = list(set(file_codes)); file_codes.sort()
    
    folder_nm = folder.split('\\')[-2]
    
    ## Make the bsc and/or the ext folder(s) and move the files to the respective folders ##
    if '002' in file_codes:
        
        opt_folder = os.path.join(folder, f'{folder_nm}/ext/')
        if not os.path.exists(opt_folder):
            os.mkdir(opt_folder)
        
        opt_files = os.path.join(folder, '*/*_002_*')
        opt_files = glob.glob(opt_files); opt_files.sort()
        for file in opt_files:
            shutil.move(file, opt_folder)
    
    if '000' in file_codes or '003' in file_codes:
        
        opt_folder = os.path.join(folder, f'{folder_nm}/bsc/')
        if not os.path.exists(opt_folder):
            os.mkdir(opt_folder)
        
        if '000' in file_codes:
            opt_files = os.path.join(folder, '*/*_000_*')
            opt_files = glob.glob(opt_files); opt_files.sort()
            for file in opt_files:
                shutil.move(file, opt_folder)
        
        if '003' in file_codes:
            opt_files = os.path.join(folder, '*/*_003_*')
            opt_files = glob.glob(opt_files); opt_files.sort()
            for file in opt_files:
                shutil.move(file, opt_folder)
    
    # sys.exit()