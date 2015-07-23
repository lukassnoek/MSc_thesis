# -*- coding: utf-8 -*-
"""
Module with functions to create .bfsl files from Presentation-logfiles
"""

import numpy as np
import os
import pandas as pd

con_names = ['Poschar','Negchar','Neuchar',
             'Posloc','Negloc','Neuloc',
             'Eval','Response']
             
con_codes = [[10],[11],[12],
             [20],[21],[22],
             [101,200],[1,9]]


def 

file_dir = '/media/lukas/Data/DynamicAffect/'
test_log = os.path.join(file_dir,'da01-DynAff_pretest_vs123123.log')

df = pd.read_table(test_log, sep = '\t',skiprows=3, 
                  header = 0, 
                  skip_blank_lines=True)

df = df.convert_objects(convert_numeric = True)
pulse_idx = int((np.where(df['Code'] == 100)[0]))

df = df.drop(range(pulse_idx))

pulse_t = df['Time'][df['Code'] == 100]

for i,code in enumerate(con_codes):
    to_write = pd.DataFrame()

    if len(code) > 1:
        idx = df['Code'].isin(xrange(code[0],code[1]+1))
    else:
        idx = df['Code'] == code
    
    to_write['Time'] = (df['Time'][idx]-float(pulse_t))/10000.0
    to_write['Duration'] = df['Duration'][idx] / 10000.0
    to_write['Weight'] = np.ones((sum(idx),1))     
    
    if univar_design:    
        to_write.to_csv(con_names[i] + '.bfsl', sep = '\t', 
                        header = False, index = False)
    elif singletrial_design:
        
        for j in xrange(to_write.shape[0]):        
            to_write2 = to_write.iloc[[j]]
            to_write2.to_csv(con_names[j] + '_00' + 
                            str(i+1) + '.bfsl', sep = '\t', 
                        header = False, index = False)
    