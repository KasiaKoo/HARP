from trace_anal import * 
import numpy as np 
import matplotlib.pyplot as plt 
""" Defining test signal and background"""
#assuming random noise in both
bg = np.random.randint(0,50, (400,2048), dtype = 'int64')
signal = np.random.randint(0,50, (400,2048), dtype = 'int64')
#adding signal
signal[250:300, 400:425] = 230*np.ones((100-50,425-400))
signal[220:280, 1400:1525] = 180*np.ones((80-20,1525-1400))
signal[250:300, 1900:1955] = 100*np.ones((100-50,1955-1900))
signal_pix = np.copy(signal)
#adding random dead pixels
for i in range(100):
    signal_pix[np.random.randint(0,400), np.random.randint(0,2048)] = 10000


art_sig = False
sig_tiff = True
file = '../../Data/vuv_70000_CROPPED.tiff'


har = HarmonicTrace()
har.set_MCPpos(30000)

if art_sig == True:
    har.load_data_array(signal)
elif sig_tiff ==True:
    har.load_data_tiff(tifffile=file)

har.get_background(bg_lim = [0,0,5,5])
har.substract_bg()
har.find_lineout()
har.define_peaks()




# plt.figure()
# plt.plot(har.lineout, alpha = 0.8)
# plt.plot(har.peaks, har.lineout[har.peaks], 'o')

# plt.xlim(1200,2000)
# plt.legend()
# plt.show()
# fig, ax = plt.subplots(2, 2)
# im0 = ax[0][0].imshow(har.data_og)
# im1 = ax[0][1].imshow(har.data)
# fig.colorbar(im0, ax = ax[0][0])
# fig.colorbar(im1, ax = ax[0][1])
# ax[0][0].set_title('bg_extracted')

# har1 = HarmonicTrace()
# har1.set_MCPpos(30000)
# har1.load_data_array(signal)
# har1.get_background_array(bg)
# har1.substract_bg()


# im3 = ax[1][0].imshow(har1.data_og)
# im4 = ax[1][1].imshow(har1.data)
# fig.colorbar(im3, ax = ax[1][0])
# fig.colorbar(im4, ax = ax[1][1])
# ax[1][0].set_title('bg from array')
# # plt.show()


# har_pix = HarmonicTrace()
# har_pix.set_MCPpos(0)
# har_pix.load_data_array(signal_pix)
# har_pix.get_background_array(bg)
# har_pix.substract_bg()

# fig,ax = plt.subplots(3)
# ax[0].imshow(har_pix.data)
# ax[0].set_title('Normal')
# har_pix.remove_dead_pix()
# ax[1].imshow(har_pix.data)
# ax[1].set_title('No dead Pixels')
# har_pix.improve_contrast(0.8)
# ax[2].imshow(har_pix.data)
# ax[2].set_title('Gamma 0.8')
# #plt.show()

# def test_spec_calib(pix_no, Mcp_pos):
#     wl = (np.arange(pix_no) - Mcp_pos+100)*0.07
#     return wl

# har_pix.find_lineout()
# har_pix.specify_spectrometer_calib(test_spec_calib)
# fig, ax = plt.subplots(2)
# ax[0].imshow(har_pix.data)
# ax[1].plot(har_pix.eV_axis, har_pix.lineout)

# har_pix.set_eVlim(5,35)
# har_pix.find_lineout()
# fig, ax = plt.subplots(2)
# ax[0].set_title('Mask?')
# ax[0].imshow(har_pix.data)
# ax[1].plot(har_pix.eV_axis, har_pix.lineout)
# #plt.show()

# fig, ax = plt.subplots(1)
# im, ax, ax2 = har_pix.plot_nice_trace(ax)
# fig.colorbar(im, ax=ax)
# plt.show()
