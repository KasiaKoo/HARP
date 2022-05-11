from PIL import Image
import os
import numpy as np
from tqdm import tqdm

#harddrive
# folder_list = [ '/Volumes/KasiaDrive/Data/20220407/Scan_MCP1850V_1cm_LaserPower350atmax_SiO2400um_repeat', '/Volumes/KasiaDrive/Data/20220408/Scan_MCP1850V_1cm_LaserPower350atmax_SiO2400um']
#solid
recent = '/Volumes/KasiaDrive/Data/20220426'
folder_list = [d for d in os.listdir(recent) if 'Scan' in d]
print(folder_list)

for f in folder_list:
    f_full = os.path.join(recent, f)
    print('enetring folder {}'.format(f_full.split('/')[-1]))
    new_folder = f_full +'_npy'
    os.makedirs(new_folder)
    print(new_folder)    
    bmps = [i for i in os.listdir(f_full) if 'bmp' in i]
    for file in tqdm(bmps):
        pic = Image.open(os.path.join(f_full,file))
        data = np.array(pic).astype('float32')
        name = '{}.npy'.format(file.split('.')[0]) 
        np.save(os.path.join(new_folder,name), data)


    



