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
import sys

class Scan():

    def __init__(self):
        self.scan_data = pd.DataFrame(columns = ['Data', 'iris', 'wedge', 'rotation', 'MCP Pos'])
        self.scan_path = ''
        self.scan_exclude = []
        self.scan_params = {'iris':None,
                            'wedge':None, 
                            'rotation':None, 
                            'MCP Pos':None,
                            'lens':0}
        self.folder = ''
        self.eV_lim = (0,500)
        self.bg_lim = [0,500,10,10]
        self.bg_array= None
        self.ver_lim = (None, None)
        self.unmasked_data = pd.DataFrame(columns = ['Data', 'iris', 'wedge', 'rotation', 'MCP Pos'])
        self.bump = []
        self.wl0 = 1800e-9

    def exclude_manual(self, file):
        self.scan_exclude.append(file)

    def set_folder(self, folder):
        self.folder = folder

    def set_params(self, iris=None, wedge=None, rotation=None, MCPPos=None, wl0=1800e-9):
        self.scan_params['iris'] = iris
        self.scan_params['wedge'] = wedge
        self.scan_params['rotation'] = rotation
        self.scan_params['MCP Pos'] = MCPPos
        self.wl0 = wl0

    def set_eVlim(self, ev_lim):
        self.eV_lim = ev_lim

    def set_verlim(self, low, up):
        self.ver_lim = (low, up)  

    """ Add no laser background"""

    def add_background(self, bg_array):
        self.bg_array = bg_array

    def add_background_from_scan(self, closed_iris_position = -45):
        bg_series = self.scan_data[self.scan_data['iris']==closed_iris_position].copy(deep=True)
        bg_series['val'] =  bg_series.apply(lambda row: row.Data.data, axis=1)
        self.bg_array = np.median(np.stack(bg_series['val'].to_numpy()), axis=0)

    def substract_bg(self, byitself=False, bg_lim = [0,0,10,10]):
        if byitself == True:
            self.scan_data['Data'] = self.scan_data['Data'].apply(lambda row: row.get_background(bg_lim))
        else:
            self.scan_data['Data'] = self.scan_data['Data'].apply(lambda row: row.get_background_array(self.bg_array))
        self.scan_data['Data'] = self.scan_data['Data'].apply(lambda row: row.substract_bg())



    """ Different ways to make populate scans with data"""

    def populate_scan_OOLI_bmp(self, stage_dict, function):
        flat_lists = [[item for item in stage_dict[key]] for key in list(stage_dict.keys())]
        combinations = list(product(*flat_lists))
        stages = list(stage_dict.keys())
        count = 0
        for i in tqdm(range(len(combinations))):
            file = os.path.join(self.folder,'{}.bmp'.format(count))
            count +=1 
            tr_temp = HarmonicTrace()
            tr_temp.set_verlim(self.ver_lim[0], self.ver_lim[1])
            tr_temp.set_wl0(self.wl0)
            tr_temp.load_data_bmp(file)
            tr_temp.get_background(bg_lim = self.bg_lim)
            tr_temp.set_MCPpos(self.scan_params['MCP Pos']) 
            tr_temp.specify_spectrometer_calib(function)
            # tr_temp.set_eVlim(self.eV_lim[0], self.eV_lim[1])
            temp_dict = self.scan_params 
            temp_dict.update({key:value for (key,value) in zip(stages,combinations[i])})
            temp_dict['Data'] = tr_temp
            self.scan_data = self.scan_data.append(temp_dict,ignore_index=True)

    def populate_scan_OOLI(self, stage_dict, function):
        flat_lists = [[item for item in stage_dict[key]] for key in list(stage_dict.keys())]
        combinations = list(product(*flat_lists))
        stages = list(stage_dict.keys())
        count = 0
        for i in tqdm(range(len(combinations))):
            file = os.path.join(self.folder,'{}.npy'.format(count))
            count +=1 
            tr_temp = HarmonicTrace()
            tr_temp.set_verlim(self.ver_lim[0], self.ver_lim[1])
            tr_temp.set_wl0(self.wl0)
            tr_temp.load_data_npy(file)
            tr_temp.get_background(bg_lim = self.bg_lim)
            tr_temp.set_MCPpos(self.scan_params['MCP Pos']) 
            tr_temp.specify_spectrometer_calib(function)
            # tr_temp.set_eVlim(self.eV_lim[0], self.eV_lim[1])
            temp_dict = self.scan_params 
            temp_dict.update({key:value for (key,value) in zip(stages,combinations[i])})
            temp_dict['Data'] = tr_temp
            self.scan_data = self.scan_data.append(temp_dict,ignore_index=True)

    def populate_scan_manual(self, files, variables, stage, function):
        for i in tqdm(range(len(files))):
            file = os.path.join(self.folder,files[i])
            step = variables[i]
            tr_temp = HarmonicTrace()
            tr_temp.set_verlim(self.ver_lim[0], self.ver_lim[1])
            tr_temp.set_wl0(self.wl0)
            tr_temp.load_data_tiff(file)
            tr_temp.get_background(bg_lim = self.bg_lim)
            if stage != 'MCP Pos':
                tr_temp.set_MCPpos(self.scan_params['MCP Pos'])
            else: 
                tr_temp.set_MCPpos(step)
            tr_temp.specify_spectrometer_calib(function)
            tr_temp.set_eVlim(self.eV_lim[0], self.eV_lim[1])
            temp_dict = self.scan_params 
            temp_dict[stage] = step
            temp_dict['Data'] = tr_temp
            self.scan_data = self.scan_data.append(temp_dict,ignore_index=True)

    def populate_scan_h5(self,h5_file, function):
        fps = glob.glob(os.path.join(self.folder, h5_file))
        for idx, fp in enumerate(fps):
            with h5.File(fp, 'r') as f:
                stages = np.array(f['Settings']['Axes']['Names'])
                positions = np.array(f['Settings']['Axes']['Positions'])
                data_t = np.array(f['Data'][:,:,:])
                data_t = data_t[:,:,:]
                if idx == 0:
                    data= np.zeros((len(fps),*data_t.shape))
                
                data[idx]=data_t
        data = data.sum(axis=0)
        positions_combo = list(product(*positions['Positions'][::-1]))
        stages = [i.decode('utf-8').lower() for i in stages]
        
        for d in tqdm(range(data.shape[0])):
            tr_temp = HarmonicTrace()
            tr_temp.set_verlim(self.ver_lim[0], self.ver_lim[1])
            tr_temp.set_wl0(self.wl0)
            tr_temp.load_data_array(data[d])
            tr_temp.get_background(bg_lim = self.bg_lim)
            tr_temp.set_MCPpos(self.scan_params['MCP Pos'])
            tr_temp.specify_spectrometer_calib(function)
            tr_temp.set_eVlim(self.eV_lim[0], self.eV_lim[1])
            temp_dict = self.scan_params.copy()
            #set different stages
            for s in range(len(stages)):
                stage = stages[s]
                temp_dict[stage] = positions_combo[d][::-1][s]
            temp_dict['Data'] = tr_temp
            self.scan_data = self.scan_data.append(temp_dict,ignore_index=True)


    def populate_scan_h5_bypart(self,h5_file, function, bypart=(0,-1):
        fps = glob.glob(os.path.join(self.folder, h5_file))
        for idx, fp in enumerate(fps):
            with h5.File(fp, 'r') as f:
                stages = np.array(f['Settings']['Axes']['Names'])
                positions = np.array(f['Settings']['Axes']['Positions'])
                data_t = np.array(f['Data'][bypart[0]:bypart[1],:,:])
                data_t = data_t[:,:,:]
                if idx == 0:
                    data= np.zeros((len(fps),*data_t.shape))
                
                data[idx]=data_t
        data = data.sum(axis=0)
        positions_combo = list(product(*positions['Positions'][::-1]))
        stages = [i.decode('utf-8').lower() for i in stages]
        
        for d in tqdm(range(data.shape[0])):
            tr_temp = HarmonicTrace()
            tr_temp.set_verlim(self.ver_lim[0], self.ver_lim[1])
            tr_temp.set_wl0(self.wl0)
            tr_temp.load_data_array(data[d])
            tr_temp.get_background(bg_lim = self.bg_lim)
            tr_temp.set_MCPpos(self.scan_params['MCP Pos'])
            tr_temp.specify_spectrometer_calib(function)
            tr_temp.set_eVlim(self.eV_lim[0], self.eV_lim[1])
            temp_dict = self.scan_params.copy()
            #set different stages
            for s in range(len(stages)):
                stage = stages[s]
                temp_dict[stage] = positions_combo[d][::-1][s]
            temp_dict['Data'] = tr_temp
            self.scan_data = self.scan_data.append(temp_dict,ignore_index=True)
        print('loaded from {} to {} out of {}'.format(bypart[0], bypart[1], len(positions_combo)))

    def add_calibration_stage(self, name, function, stage):
        self.scan_data[name] = self.scan_data[stage].apply(function).round(2)

    
    def save_npz(self, save_folder, save_name):
        test1 = self.scan_data.copy()
        test1['data'] = test1.apply(lambda row: row.Data.data, axis=1)
        test1['eV'] = test1.apply(lambda row: row.Data.eV_axis, axis=1)
        test1['ver'] = test1.apply(lambda row: row.Data.ver_lim, axis=1)
        test1['wl0'] = test1.apply(lambda row: row.Data.wl0, axis=1)
        test1 = test1.drop(['Data'], axis=1)
        np.savez(os.path.join(save_folder, save_name), 
                iris = test1['iris'].values, 
                intensity = test1['intensity'].values,
                wedge=test1['wedge'].values,
                rotation=test1['rotation'].values,
                lens=test1['lens'].values,
                mcppos=test1['MCP Pos'].values, 
                data=test1['data'].values,
                ver=test1['ver'].values,
                wl0=test1['wl0'].values,
                eV = test1['eV'].values)

    def load_npz(self,save_folder, save_name):
        df = pd.DataFrame()
        with np.load(os.path.join(save_folder, save_name), allow_pickle=True) as f:
            df['iris'] = f['iris'] 
            df['intensity'] = f['intensity']
            df['wedge'] = f['wedge']
            df['rotation'] = f['rotation']
            df['lens'] = f['lens']
            df['MCP Pos'] = f['mcppos'] 
            df['data'] = f['data']
            df['eV'] = f['eV']
            df['ver'] = f['ver']
            df['wl0'] = f['wl0']
        return df

    def sort_by_stage(self, stage):
        self.scan_data = self.scan_data.sort_values(by=stage)

    
    def mask_data(self, mask):
        self.unmasked_data = self.scan_data.copy(deep=True)
        self.scan_data = self.scan_data[mask]

    def revert_mask(self):
        self.scan_data = self.unmasked_data

    def gaus(self, x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def find_lineout(self, array):
        p0 = [max(array.sum(1)), array.shape[0]/2, array.shape[0]/4]
        # ver_lineout = array.sum(axis=1)-min(array.sum(axis=1)) 
        # popt, pcov = curve_fit(self.gaus, range(array.shape[0]), ver_lineout, p0)
        # y0 = round(int(popt[1]))
        # sig = round(abs(int(popt[2])))
        lineout = array.sum(axis=0)
        return lineout - min(lineout)

    def return_scan_data(self, stage, df = False):
        if type(df) == bool:
            df = self.scan_data
        temp_df = df.copy(deep=True)
        temp_df['val'] = temp_df.apply(lambda row: row.Data.data, axis=1)
        ave = temp_df.groupby(stage)['val'].apply(lambda x: np.mean(x, axis=0))
        x = ave.index.values
        Z = np.array([self.find_lineout(i) for i in ave.values])
        if len(self.bump)!=0:
            Z = Z-self.bump
            Z[Z<0]=0
        y = self.scan_data['Data'][0].eV_axis
        return x, y, Z


    def return_scan_data_slab(self, stage, df = False):
        if type(df) == bool:
            df = self.scan_data
        temp_df = df.copy(deep=True)
        temp_df['val'] = temp_df.apply(lambda row: row.Data.data, axis=1)

        ave = temp_df.groupby(stage)['val'].apply(lambda x: np.mean(x, axis=0))
        x = ave.index.values
        ave = temp_df.groupby(stage)['val'].apply(lambda x: np.mean(x, axis=0))
        Z = np.stack(ave.values)
        x = x.reshape(x.shape[0],1)*np.ones([x.shape[0],Z.shape[1]])
        Z = Z.reshape(Z.shape[0]*Z.shape[1], Z.shape[2])
        x = x.reshape(Z.shape[0])
        y = self.scan_data['Data'][0].nm_axis
        return x, y, Z

    #Plotting Functions
        
    def plot_average_log(self, ax, mask=None):
        """ Does plot average countour so not good for different eV"""
        temp_df = self.scan_data.copy(deep=True)
        if mask ==None:
            mask = temp_df ==temp_df
        temp_list = temp_df[mask]['Data'].tolist()
        data = np.array([i.data for i in temp_list])
        data_ave = data.mean(0)
        im = ax.contourf(temp_list[0].eV_axis, range(data_ave.shape[0]),np.log(data_ave), levels=50,cmap='magma', aspect = 'auto')
        ax.set_xlabel('Signal Energy eV')
        ax2 = ax.twinx()
        lo =  self.find_lineout(data_ave)
        if len(self.bump)!=0:
            lo = lo - self.bump
            lo[lo<0]=0
        ax2.plot(temp_list[0].eV_axis, lo, color='orange')
        ax2.fill_between(temp_list[0].eV_axis, lo, alpha = 0.4, color='orange')
        ax2.set_ylim(bottom=np.min(lo), top = 3*np.max(lo))
        ax2.set_yticks([])
        return im, ax, ax2

    def plot_average(self, ax, mask = None):
        """ Does plot average countour so not good for different eV"""
        temp_df = self.scan_data.copy(deep=True)
        if mask ==None:
            mask = temp_df ==temp_df
        temp_list = temp_df[mask]['Data'].tolist()
        data = np.array([i.data for i in temp_list])
        data_ave = data.mean(0)
        im = ax.contourf(temp_list[0].eV_axis, range(data_ave.shape[0]),data_ave, levels=50,cmap='magma', aspect = 'auto')
        ax.set_xlabel('Signal Energy eV')
        ax2 = ax.twinx()
        lo =  self.find_lineout(data_ave)
        if len(self.bump)!=0:
            lo = lo - self.bump
            lo[lo<0]=0
        ax2.plot(temp_list[0].eV_axis, lo, color='orange')
        ax2.fill_between(temp_list[0].eV_axis, lo, alpha = 0.4, color='orange')
        ax2.set_ylim(bottom=np.min(lo), top = 3*np.max(lo))
        ax2.set_yticks([])
        return im, ax, ax2

    def plot_scan_mean(self, ax, stage, mask=None):
        temp_df = self.scan_data.copy(deep=True)
        if mask ==None:
            mask = temp_df ==temp_df
        temp_df['val'] = temp_df[mask].apply(lambda row: row.Data.data, axis=1)
        ave = temp_df.groupby(stage)['val'].apply(lambda x: np.mean(x, axis=0))
        x = ave.index.values
        Z = np.array([self.find_lineout(i) for i in ave.values])
        if len(self.bump)!=0:
            Z = Z-self.bump
            Z[Z<0]=0
        y = self.scan_data['Data'][0].eV_axis
        im = ax.contourf(x, y, Z.T, levels=100, cmap = 'magma', aspect = 'auto')
        return im, ax

    def plot_scan_mean_log(self, ax, stage, mask=None):
        temp_df = self.scan_data.copy(deep=True)
        if mask ==None:
            mask = temp_df ==temp_df
        temp_df['val'] = temp_df[mask].apply(lambda row: row.Data.data, axis=1)
        ave = temp_df.groupby(stage)['val'].apply(lambda x: np.mean(x, axis=0))
        x = ave.index.values
        Z = np.array([self.find_lineout(i) for i in ave.values])
        if len(self.bump)!=0:
            Z = Z-self.bump
            Z[Z<0]=0
        y = self.scan_data['Data'][0].eV_axis
        im = ax.contourf(x, y, np.log(Z.T), levels=100, cmap = 'magma', aspect = 'auto')
        return im, ax 

    def plot_lineouts_mean(self, ax, stage, no_plots, mask=None):
        temp_df = self.scan_data.copy(deep=True)
        if mask ==None:
            mask = temp_df ==temp_df
        temp_df['val'] = temp_df[mask].apply(lambda row: row.Data.data, axis=1)
        ave = temp_df.groupby(stage)['val'].apply(lambda x: np.mean(x, axis=0))
        x = ave.index.values
        Z = np.array([self.find_lineout(i) for i in ave.values])
        if len(self.bump)!=0:
            Z = Z-self.bump
            Z[Z<0]=0
        y = self.scan_data['Data'][0].eV_axis
        idx_plot = [i for i in range(len(x)) if i%(len(x)//no_plots)==0]
        cmap = plt.get_cmap('magma')
        colors = [cmap(i) for i in np.linspace(0, 1, len(idx_plot)+1)]
        # ax.set_prop_cycle(color = colors)
        count = 0
        for i in idx_plot:
            ax[count].plot(y, Z[i] - min(Z[i]), label = x[i], lw=2)
            ax[count].set_ylabel(str(x[i])) 
            ax[count].get_yaxis().set_ticks([])
            count += 1
            # ax.fill_between(y, Z[i] - min(Z[i]), alpha = 0.6)
        return ax

    def plot_lineouts_mean_all(self, ax, stage, mask=None):
        temp_df = self.scan_data.copy(deep=True)
        if mask ==None:
            mask = temp_df ==temp_df
        temp_df['val'] = temp_df[mask].apply(lambda row: row.Data.data, axis=1)
        ave = temp_df.groupby(stage)['val'].apply(lambda x: np.mean(x, axis=0))
        x = ave.index.values
        Z = np.array([self.find_lineout(i) for i in ave.values])
        if len(self.bump)!=0:
            Z = Z-self.bump
            Z[Z<0]=0
        y = self.scan_data['Data'][0].eV_axis
        cmap = plt.get_cmap('magma')
        colors = [cmap(i) for i in np.linspace(0, 1, len(x)+1)]
        ax.set_prop_cycle(color = colors)
        for i in range(len(x)):
            ax.plot(y, Z[i] - min(Z[i]), label = x[i], lw=2)
            ax.get_yaxis().set_ticks([])
        return ax

    def plot_lineouts_mean_all_log(self, ax, stage, mask = None):
        temp_df = self.scan_data.copy(deep=True)
        if mask ==None:
            mask = temp_df ==temp_df
        temp_df['val'] = temp_df[mask].apply(lambda row: row.Data.data, axis=1)
        ave = temp_df.groupby(stage)['val'].apply(lambda x: np.mean(x, axis=0))
        x = ave.index.values
        Z = np.array([self.find_lineout(i) for i in ave.values])
        if len(self.bump)!=0:
            Z = Z-self.bump
            Z[Z<0]=0
        y = self.scan_data['Data'][0].eV_axis
        cmap = plt.get_cmap('magma')
        colors = [cmap(i) for i in np.linspace(0, 1, len(x)+1)]
        ax.set_prop_cycle(color = colors)
        for i in range(len(x)):
            ax.plot(y, np.log(Z[i] - min(Z[i])), label = x[i], lw=2)
            ax.get_yaxis().set_ticks([])
        return ax

    def plot_scan_sum(self, ax, stage, mask=None):
        temp_df = self.scan_data.copy(deep=True)
        if mask ==None:
            mask = temp_df ==temp_df
        temp_df['val'] = temp_df[mask].apply(lambda row: row.Data.data, axis=1)
        ave = temp_df.groupby(stage)['val'].apply(lambda x: np.sum(x, axis=0))
        x = ave.index.values
        Z = np.array([self.find_lineout(i) for i in ave.values])
        if len(self.bump)!=0:
            Z = Z-self.bump
            Z[Z<0]=0
        y = self.scan_data['Data'][0].eV_axis
        im = ax.contourf(x, y, Z.T, levels=100, cmap = 'magma', aspect='auto')
        return im, ax

    def plot_polar_scan(self, ax, mask=None):
        stage = 'rotation'
        temp_df = self.scan_data.copy(deep=True)
        if mask ==None:
            mask = temp_df ==temp_df
        temp_df['val'] = temp_df[mask].apply(lambda row: row.Data.data, axis=1)
        ave = temp_df.groupby(stage)['val'].apply(lambda x: np.mean(x, axis=0))
        x = ave.index.values
        Z = np.array([self.find_lineout(i) for i in ave.values])
        if len(self.bump)!=0:
            Z = Z-self.bump
            Z[Z<0]=0
        y = self.scan_data['Data'][0].eV_axis
        r, theta = np.meshgrid(y, np.radians(x))
        im = ax.contourf(theta,r, Z, levels=100, cmap = 'magma', aspect='auto')
        ax.set_thetalim(np.radians(min(x)), np.radians(max(x)))
        return im, ax

    def plot_polar_scan_log(self, ax, mask = None):
        stage = 'rotation'
        temp_df = self.scan_data.copy(deep=True)
        if mask ==None:
            mask = temp_df ==temp_df
        temp_df['val'] = temp_df[mask].apply(lambda row: row.Data.data, axis=1)
        ave = temp_df.groupby(stage)['val'].apply(lambda x: np.mean(x, axis=0))
        x = ave.index.values
        Z = np.array([self.find_lineout(i) for i in ave.values])
        if len(self.bump)!=0:
            Z = Z-self.bump
            Z[Z<0]=0
        y = self.scan_data['Data'][0].eV_axis
        r, theta = np.meshgrid(y, np.radians(x))
        im = ax.contourf(theta,r, np.log(Z), levels=100, cmap = 'magma', aspect='auto')
        ax.set_thetalim(np.radians(min(x)), np.radians(max(x)))
        return im, ax



        



        
        

        












