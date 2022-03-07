from context import Harmonics_Analysis
from Harmonics_Analysis.scan_anal import *
from Harmonics_Analysis.iris_functions import Iris
import os
from scipy.optimize import curve_fit
import warnings
from matplotlib import ticker
warnings.filterwarnings('ignore')
def pos_to_off(position_value, popt = [1.26493902e-02,-1.25115854e+03]):
    offset_val = popt[0]*position_value +  popt[1]
    return offset_val

def Andor_calib(pix_no, trans_pos=30000):
    coeff = [-0.0702247191011236,1240/24.8]#[-5.29194209e-02,  56.6780912]
    offset = 1281
    trans_pos_offset = pos_to_off(trans_pos)
    wl = (np.arange(pix_no)+trans_pos_offset-offset) * coeff[0] + coeff[1]
    return wl 
#Iris calibration from 28th Jan at 1800 nm short pulse
""" 1800 nm scans"""
power_2801_1800short = np.array([184, 180, 180,180,180,180,176, 180,180,165,162,150,130,110,86,58,31,11.5,4])*1e-3
iris_pos_2801_1800short = np.linspace(45,-45,19)
iris_2801 = Iris()
iris_2801.specify_calib(iris_positions=iris_pos_2801_1800short, powers=power_2801_1800short)
iris_2801.specify_params(w0_init=100e-6, f=0.75, wl=1800e-9, M2=1, reprate=1000,pulse_duration=15e-15)
test = Scan()
test.set_verlim(None, None)
test.set_eVlim((5,35))
test_manual = False
test_h5=True

if test_manual == True:
    folder = '../../Data/2022-01-28/Iris_scan_SampleX-10_Y-5.25_Z6.8_rot_2_MgO_SWIRMCP1200V_scanningDown_MCPpos70'
    files = [i for i in os.listdir(folder)]
    exclude = ['-29_120secexp.tif', '-30_120secexp.tif']
    files = [i for i in files if i not in exclude]
    variables = [float(i[:-4]) for i in files]
    test.set_params(rotation=0, MCPPos=70000,wedge=1060)
    test.populate_scan_manual(files, variables, stage='iris', function = Andor_calib)


elif test_h5 == True:
    test.set_folder('../../Data')
    h5_file = 'ImgData-000-20220201-171855.h5'
    test.set_params(wedge=1060, MCPPos=70000)
    test.populate_scan_h5(h5_file, function = Andor_calib)




test.add_calibration_stage('intensity', iris_2801.get_intensity_TWcm2, 'iris')
stage = 'rotation'
test.sort_by_stage(stage)

test.add_background_from_scan(closed_iris_position=-10)
test.substract_bg()
fig, ax = plt.subplots(1)
ax = test.plot_average(ax)
plt.show()

mask = test.scan_data['iris']<=-7
test.mask_data(mask)
test.define_bump((None,1))
x,y,Z = test.return_scan_data(stage)

test_plot = False
if test_plot == True: 
    fig, ax = plt.subplots(2, sharex = True)
    ax[0] = test.plot_lineouts_mean_all(ax[0], stage)
    test.revert_mask()

    test.define_bump((None,2))
    mask = test.scan_data['iris']<=-28
    test.mask_data(mask)

    ax[1] = test.plot_lineouts_mean_all(ax[1], stage)
    ax[1].set_xlabel('Energy eV')
    plt.show()

    fig, ax = plt.subplots(2)
    im, ax[0] = test.plot_scan_mean(ax[0], stage)
    im, ax[1] = test.plot_scan_mean_log(ax[1], stage)
    plt.show()

    no_plot = 5
    fig, ax = plt.subplots(no_plot, sharex=True)
    ax = test.plot_lineouts_mean(ax, stage, no_plot-1)
    ax[-1].set_xlabel('Energy eV')
    plt.show()

