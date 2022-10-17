import matplotlib.pyplot as plt
import numpy as np 
import os
from PIL import Image
from matplotlib.widgets import Slider
import pandas as pd
from tqdm import tqdm
path = '/Volumes/KasiaDrive/Data/20220729'

data = pd.DataFrame(columns = ['wavelength', 'Gas', 'MCP Voltage', 'Exposure time', 'MCP height', 'filter', 'repeat', 'pic'])
#choosing files
files = os.listdir(path)
for file in tqdm(files):
    image = np.array(Image.open(os.path.join(path, file)))
    file = file[:-4]
    wl, g, V, t, h, f, r = file.split('_')
    dic = {'wavelength':wl, 'Gas':g, 'MCP Voltage':V,  'Exposure time':t, 'MCP height':h[6:-2], 'filter':f, 'repeat':r, 'pic':image}
    print(dic)
    data = data.append(dic, ignore_index=True)



