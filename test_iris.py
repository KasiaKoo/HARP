import numpy as np 
from context import Harmonics_Analysis
from Harmonics_Analysis.iris_functions import Iris
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter 
from scipy.optimize import curve_fit
import pandas as pd 

def gaus(x, a, x0, sig):
    return a * np.exp(-(x-x0)**2/(2*sig**2))
iris_position = np.array([388,385,380 ,375  ,370 ,365  ,360,355  ,350  ,345  ,340  ,335  ,330])
power =np.array([ 0 ,5 ,8 ,37 ,112 ,177 ,230 ,260 ,290 ,300 ,310 ,315 ,312])*1e-3
spectrum_file = "~/Documents/Data/2202_idler_spectrum.txt"
df = pd.read_csv(spectrum_file)
wl = df['Wavelength [nm]'].to_numpy()
f = 3e8/(wl*1e-9)
I = df['Intensity2'].to_numpy()
I = savgol_filter(I, 11,3)
I = I[200:]
f = f[200:]
N = len(f)
del_f = abs(f[1]-f[0])
t = fftfreq(N, del_f)
It =  np.abs(fft(I))
p0 = (max(It), 0, 20e-15)
popt, pcov = curve_fit(gaus, t, It, p0=p0)
# fig, ax = plt.subplots(2)
# ax[0].plot(wl[200:], I)
# ax[0].set_xlabel('Wavelength [nm]')
# ax[1].scatter(t, It)
# ax[1].plot(t, gaus(t, *popt))
# ax[1].set_xlabel('Time [s]')
# ax[1].set_xlim(-0.5e-12, 0.5e-12)
# plt.show()
FWHM = 2*np.sqrt(2*np.log(2))*popt[-1]
iris = Iris()
iris.specify_calib(iris_positions=iris_position, powers=power)
iris.specify_params(w0_init=100e-6, f=0.75, wl=1800e-9, M2=1, reprate=1000,pulse_duration=15e-15)

plt.plot(np.linspace(320,390,100), iris.get_intensity_TWcm2(np.linspace(320,390,100)))
plt.xlabel(iris_position)
plt.xlim(360, 390)
plt.show()

""" Oli's COde
import numpy as np
import matplotlib.pyplot as plt
gauss = lambda x, s: np.exp(-0.5*((x/s)**2))

t = np.linspace(-1,1,1000)
I = gauss(t, 0.1)


com = np.average(t, weights=I)
sd = np.sqrt(np.average((t-com)**2, weights=I))

print(sd)

plt.figure()
plt.plot(t,I)

plt.show()

"""
