import matplotlib.pyplot as plt
import numpy as np 
import os
from PIL import Image
from matplotlib.widgets import Slider

path = '/Volumes/KasiaDrive/Data/20220729'


#choosing files
files = os.listdir(path)
wl_mask = np.array([f[0]=='8' for f in files])
fl_mask = np.array(['empty' in f for f in files])
repeat_mask = np.array([f[-5:]=='1.bmp' for f in files])
gas_mask = np.array(['Argon' in f for f in files])
time_mask = np.array([f[28:33]=='10000' in f for f in files])
mask = wl_mask*fl_mask*repeat_mask*gas_mask*(~time_mask)
files_800 = np.array(files)[mask]
heigh_mask = np.array([f[42:43]=='3' for f in files])
files_filter = np.array(files)[heigh_mask*wl_mask*repeat_mask*gas_mask]

# print(files_800)
mcp_low = 1000
mcp_high = 2500
im = Image.open(os.path.join(path,files_800[0]))
p = np.array(im)

# fig, ax = plt.subplots(2)
# p = ax[1].imshow(im, vmax = 200, cmap = 'nipy_spectral')
# ax[1].axhspan(mcp_low,mcp_high, alpha = 0.2)
# plt.colorbar(p, ax = ax[1])

# ax[0].plot(np.sum(im,axis=1))
# ax[0].axvspan(mcp_low,mcp_high, alpha = 0.2)

# plt.show()


# for i in files_800:
#     height = int(i[-16:-15])
#     im = Image.open(os.path.join(path, i))
#     p = np.array(im)
#     plt.plot(np.arange(p.shape[1]) + height*1000, np.sum(p[mcp_low:mcp_high], axis=0), label = height)

# plt.legend()
# plt.show()



def pixel(offset, height, c1, c2,p_shape = 5468,):
    pix = np.arange(p_shape)
    return c1*(pix + offset*height) + c2

# Define initial parameters
init_offset = 1000
init_c1 = 1
init_c2 = 0

# Create the figure and the line that we will manipulate
MCP_signal_array = np.zeros((p.shape[1], len(files_800)))
height_list = []
count =0
for i in files_800:
    height_list.append(int(i[-16:-15]))
    im = Image.open(os.path.join(path, i))
    p = np.array(im)
    MCP_signal_array[:,count] = np.sum(p[mcp_low:mcp_high], axis=0)
    count = count +1

filter_signal_array = np.zeros((p.shape[1], len(files_filter)))
count =0
for i in files_filter:
    im = Image.open(os.path.join(path, i))
    p = np.array(im)
    filter_signal_array[:,count] = np.sum(p[mcp_low:mcp_high], axis=0)
    count = count +1



"""________________________Finding offsets_________________________"""
fig, ax = plt.subplots(2)
ax[0].grid()
line0, =  ax[0].plot(pixel(init_offset,0, init_c1, init_c2),MCP_signal_array[:,0]+0*100000, lw=2, alpha=0.8, label ='height 0')
line1, =  ax[0].plot(pixel(init_offset,1, init_c1, init_c2),MCP_signal_array[:,1]+1*100000, lw=2, alpha=0.8, label ='height 1')
line2, =  ax[0].plot(pixel(init_offset,2, init_c1, init_c2),MCP_signal_array[:,2]+2*100000, lw=2, alpha=0.8, label ='height 2')
line3, =  ax[0].plot(pixel(init_offset,3, init_c1, init_c2),MCP_signal_array[:,3]+3*100000, lw=2, alpha=0.8, label ='height 3')
line4, =  ax[0].plot(pixel(init_offset,4, init_c1, init_c2),MCP_signal_array[:,4]+4*100000, lw=2, alpha=0.8, label ='height 4')
line5, =  ax[0].plot(pixel(init_offset,5, init_c1, init_c2),MCP_signal_array[:,5]+5*100000, lw=2, alpha=0.8, label ='height 5')
line6, =  ax[0].plot(pixel(init_offset,6, init_c1, init_c2),MCP_signal_array[:,6]+6*100000, lw=2, alpha=0.8, label ='height 6')

ax[0].legend()
ax[0].set_xlabel('Pixel')
ax[0].set_xlim(0, 13000)

ax[1].grid()
line_empty, =  ax[1].plot(pixel(init_offset,3, init_c1, init_c2),filter_signal_array[:,0]+0*100000, lw=2, alpha=0.8, label ='empty')
line_Zr, =  ax[1].plot(pixel(init_offset,3, init_c1, init_c2),filter_signal_array[:,1]+1*100000, lw=2, alpha=0.8, label ='Zr')
line_Al, =  ax[1].plot(pixel(init_offset,3, init_c1, init_c2),filter_signal_array[:,2]+2*100000, lw=2, alpha=0.8, label ='Al')
line_p3ht, =  ax[1].plot(pixel(init_offset,3, init_c1, init_c2),filter_signal_array[:,3]+3*100000, lw=2, alpha=0.8, label ='p3ht')
ax[1].legend()

ax[0].legend()
ax[0].set_xlabel('Pixel')
ax[0].set_xlim(0, 13000)
# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axoff = plt.axes([0.25, 0.1, 0.65, 0.03])

off_slider = Slider(
    ax=axoff,
    label='Offset',
    valmin=500,
    valmax=1500,
    valinit=init_offset,
)

axc2 = plt.axes([0.1, 0.25, 0.0225, 0.63])

c2_slider = Slider(
    ax=axc2,
    label='Coefficient 2',
    valmin=-1000,
    valmax=1000,
    valinit=init_c2,
    orientation="vertical"
)
axc1 = plt.axes([0.2, 0.25, 0.0225, 0.63])

c1_slider = Slider(
    ax=axc1,
    label='Coefficient 1',
    valmin=0.1,
    valmax=2,
    valinit=init_c1,
    orientation="vertical"
)

# The function to be called anytime a slider's value changes
def update(val):
    line0.set_xdata(pixel(off_slider.val,0, c1_slider.val,c2_slider.val))
    line1.set_xdata(pixel(off_slider.val,1, c1_slider.val,c2_slider.val))
    line2.set_xdata(pixel(off_slider.val,2, c1_slider.val,c2_slider.val))
    line3.set_xdata(pixel(off_slider.val,3, c1_slider.val,c2_slider.val))
    line4.set_xdata(pixel(off_slider.val,4, c1_slider.val,c2_slider.val))
    line5.set_xdata(pixel(off_slider.val,5, c1_slider.val,c2_slider.val))
    line6.set_xdata(pixel(off_slider.val,6, c1_slider.val,c2_slider.val))
    line_empty.set_xdata(pixel(off_slider.val,3, c1_slider.val,c2_slider.val))
    line_Zr.set_xdata(pixel(off_slider.val,3, c1_slider.val,c2_slider.val))
    line_Al.set_xdata(pixel(off_slider.val,3, c1_slider.val,c2_slider.val))
    line_p3ht.set_xdata(pixel(off_slider.val,3, c1_slider.val,c2_slider.val))
    fig.canvas.draw_idle()


# # register the update function with each slider
off_slider.on_changed(update)
c1_slider.on_changed(update)
c2_slider.on_changed(update)
plt.show()



