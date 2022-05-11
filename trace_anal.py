import numpy as np
from PIL import Image
import units
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
eV_times_m = units.Unit().eV_times_m

class HarmonicTrace():
    """ Class Analysing individual picture from an MCP image"""
    def __init__(self):
        self.data = np.zeros((2,2))
        self.data_og= np.zeros((2,2))
        self.pix_axis = np.ones(2048)
        self.lineout = np.zeros(2)
        self.Spec_calib= None
        self.eV_lim = (0,200)
        self.eV_axis = np.ones(2048)
        self.nm_axis = np.ones(2048)
        self.order_axis = np.ones(2048)
        self.bg = np.zeros((2,2))
        #Parameters of the experiment
        self.MCP_pos = 0 #in units used for calibration
        self.wl0 = 1800e-9 #in m
        self.eV0 = eV_times_m/self.wl0
        self.ver_lim = (None, None)
        self.peaks = None

    def set_MCPpos(self, value=0):
        self.MCP_pos = value
        return self

    def set_verlim(self, low, high):
        self.ver_lim = (low, high)
        return self

    def load_data_array(self, array):
        #non cropped x axis!
        self.data = array[self.ver_lim[0]:self.ver_lim[1],:]
        self.data_og = array[self.ver_lim[0]:self.ver_lim[1],:]
        self.pix_axis = np.arange(self.data.shape[1])
    def crop_ver(self):
        self.data = self.data[self.ver_lim[0]:self.ver_lim[1],:]

    def load_data_npy(self, npyfile):
        array = np.load(npyfile).T
        self.data = array[self.ver_lim[0]:self.ver_lim[1],:]
        self.data_og = array[self.ver_lim[0]:self.ver_lim[1],:]
        self.pix_axis = np.arange(self.data.shape[1])
    def load_data_tiff(self, tifffile):
        pic = Image.open(tifffile)
        arr = np.array(pic).astype('float32').T
        self.data = arr[self.ver_lim[0]:self.ver_lim[1],:]
        self.data_og = arr[self.ver_lim[0]:self.ver_lim[1],:]
        self.pix_axis = np.arange(self.data.shape[1])
        return self


    def set_eVlim(self, low, high):
        self.eV_lim = (low, high)
        mask = (self.eV_axis<self.eV_lim[1])*(self.eV_axis>self.eV_lim[0])
        self.eV_axis = self.eV_axis[mask]
        self.nm_axis = self.nm_axis[mask]
        self.order_axis = self.order_axis[mask]
        self.data = self.data[:,mask]
        return self
        
    def restor_og_data(self):
        self.data = self.data_og
        return self

    def specify_spectrometer_calib(self, function):
        pix_no = len(self.pix_axis)
        self.nm_axis = function(pix_no,self.MCP_pos)
        self.eV_axis = eV_times_m*1e-9/self.nm_axis
        self.order_axis = self.eV_axis/self.eV0 
        return self


    
    def get_background_fromtiff(self, tifffile):
        pic = Image.open(tifffile)
        self.bg = np.array(pic)
        return self

    def get_background(self, bg_lim = [0,500,10,10]):
        x,y,w,h = bg_lim
        bg_val = np.median(self.data_og[y:y+h, x:x+w])
        self.bg = bg_val*np.ones(self.data.shape)
        return self

    def get_background_array(self, array):
        self.bg = array 
        return self

    def substract_bg(self):
        self.data = self.data- self.bg
        self.data[self.data<0] = 0
        return self
        
    def find_lineout(self):
        # popt, pcov = curve_fit(self.gaus, range(self.data.shape[0]), self.data.sum(axis=1)-min(self.data.sum(axis=1)))
        y0 = round(int(popt[1]))
        sig = round(abs(int(popt[2])))
        lineout = self.data[y0-sig:sig+y0,:].sum(axis=0)
        self.lineout = lineout - min(lineout)
        return self
    
    def gaus(self, x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    def define_peaks(self, minpeak=0.1, window = 11):
        self.find_lineout()
        filtered = savgol_filter(self.lineout, window, 3)
        filtered[filtered<0]=0
        peaks, _ = find_peaks(filtered, minpeak*max(filtered))
        self.peaks = peaks
        return self


    def remove_dead_pix(self):
        sorted_flt = sorted(self.data.flatten())
        rouge_pixel = np.argwhere(np.array(np.diff(sorted_flt))>max(sorted_flt)/10)
        if len(rouge_pixel) !=0:
            cutoff = sorted_flt[int(min(rouge_pixel))]
            self.data[self.data>cutoff]=0
        return self

    def improve_contrast(self, gamma):
        max_pix = max(self.data.flatten())
        self.data = ((self.data/max_pix)**gamma)*max_pix
        return self

    def plot_nice_trace(self, ax):
        im = ax.contourf(self.eV_axis, range(self.data.shape[0]), np.log(self.data), levels=100, cmap = 'magma')
        ax.set_xlabel('Energy eV')
        ax2 = ax.twinx()
        ax2.plot(self.eV_axis, self.lineout)
        #ax2.fill_between(range(n.shape[1]), n.sum(axis=0), alpha=alpha, color=linout_color)
        ax2.set_ylabel('Counts [a.u.]')
        ax2.set_ylim(bottom=0, top = 4*np.max(self.lineout))
        return im, ax, ax2

        


