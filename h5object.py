import sys, os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from trace_anal import HarmonicTrace
from tqdm import tqdm
from scipy.signal import savgol_filter
import glob
import h5py as h5
from itertools import product
from matplotlib.patches import Rectangle


class h5object():

    def __init__(self, folder = './', file = 'h5file.h5'):
        self.folder_path =folder
        self.h5file = file
        self.positions_combo = []
        self.pos_df = pd.DataFrame()
        self.masking_array = np.zeros((3,1))
        self.naming_dic = {}
        self.mask = [True]*self.masking_array.shape[1]
        self.masked_idx = self.masking_array[0,self.mask]

    def set_folder(self, folder):
        self.folder_path = folder
        
    def set_file(self, file):
        self.h5file = file
        
    def load_stages(self):
        self.fps = glob.glob(os.path.join(self.folder_path, self.h5file))
        for idx, fp in enumerate(self.fps):
            with h5.File(fp, 'r') as f:
                self.stage_names = np.array(f['Settings']['Axes']['Names'])
                self.positions = np.array(f['Settings']['Axes']['Positions'])

        self.stage_names = [i.decode('utf-8').lower() for i in self.stage_names][::-1]
    
        positions_combo = list(product(*self.positions['Positions'][::-1]))
        pos_df = pd.DataFrame(columns = self.stage_names)
        for d in range(len(positions_combo)):
            df_temp = pd.DataFrame([positions_combo[d]], columns=self.stage_names, index=[d])
            pos_df = pos_df.append(df_temp)


        masking_array = np.zeros((len(self.stage_names)+1, len(positions_combo)))
        masking_array[0,:] = pos_df.index
        count =1
        naming_dic = {}
        for stage in self.stage_names:
            masking_array[count,:] = pos_df[stage].values
            naming_dic[stage] = count
            count +=1 
        
            
        
        self.positions_combo = positions_combo
        self.pos_df = pos_df
        self.pos_df['data']=None 
        self.masking_array = masking_array
        self.mask = [True]*self.masking_array.shape[1]
        self.masked_idx = self.masking_array[0,self.mask]
        self.naming_dic = naming_dic
    
    def find_mask(self, stage_name, pos_stage):
        if type(pos_stage)==tuple:
            self.mask = (self.masking_array[self.naming_dic[stage_name]]>=pos_stage[0])*(self.masking_array[self.naming_dic[stage_name]]<=pos_stage[1])
        else:
            self.mask = self.masking_array[self.naming_dic[stage_name]]==pos_stage
        
        self.masked_idx = self.masking_array[0,self.mask]
    
    def load_data(self):
        for idx, fp in enumerate(self.fps):
            with h5.File(fp, 'r') as f:
                self.data = np.array(f['Data'][self.masked_idx,:,:])
        
        self.pos_df['data'][self.masked_idx] = list(self.data)

        
    
    
    

        
    



