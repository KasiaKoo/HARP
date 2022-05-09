from PIL import Image
import os
import numpy as np
from tqdm import tqdm

#harddrive
folder_list = [ '/Volumes/KasiaDrive/Data/20220407/Scan_MCP1850V_1cm_LaserPower350atmax_SiO2400um_repeat', '/Volumes/KasiaDrive/Data/20220408/Scan_MCP1850V_1cm_LaserPower350atmax_SiO2400um']

for f in folder_list:
    print('enetring folder {}'.format(f.split('/')[-1]))
    new_folder = f +'_npy'
    os.makedirs(new_folder)
    print(new_folder)    
    bmps = [i for i in os.listdir(f) if 'bmp' in i]
    for file in tqdm(bmps):
        pic = Image.open(os.path.join(f,file))
        data = np.array(pic).astype('float32')
        name = '{}.npy'.format(file.split('.')[0]) 
        np.save(os.path.join(new_folder,name), data)


    



