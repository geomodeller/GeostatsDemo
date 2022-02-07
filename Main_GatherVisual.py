import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize

poro_ori = np.load('Numpy_P50_porosity.npy')
perm_ori = np.load('Numpy_P50_permeability.npy')
rock_ori = np.load('Numpy_P50_RockType.npy')
tempimage = 'tempimage'
nz_fine = 30
ny_fine = 211
nx_fine = 211
nz, ny, nx = 30, 128, 128
nz_cap = 2
def Upsacling(poro_ori, perm_ori, rock_ori, nz = 8, ny = 64, nx = 64):
    poro_upscale = np.zeros((nz, ny, nx))
    perm_upscale = np.zeros((nz, ny, nx))
    rock_upscale = np.zeros((nz, ny, nx))
    Poro_cap = poro_ori[:2]
    Poro_res = poro_ori[2:]
    Perm_cap = perm_ori[:2]
    Perm_res = perm_ori[2:]
    poro_upscale[:2] = resize(Poro_cap, (1,ny, nx), anti_aliasing=False)
    poro_upscale[2:] = resize(Poro_res, (28,ny, nx), anti_aliasing=False)
    perm_upscale[:2] = resize(Perm_cap, (1,ny, nx), anti_aliasing=False)
    perm_upscale[2:] = resize(Perm_res, (28,ny, nx), anti_aliasing=False)
    rock_ori = np.array(rock_ori, dtype = float)
    rock_upscale[0] = 1
    rock_upscale[2:] = resize(rock_ori[2:], (28,ny, nx), anti_aliasing=True)
    rock_upscale[2:] [rock_upscale[2:]<2.5] = 2
    rock_upscale[2:] [rock_upscale[2:]>=2.5] = 3
    # Assign hard data
    a_, b_ = int((nx-1)/3),int(2*(nx-1)/3)
    for i, j, i_, j_ in zip([a_,a_,b_,b_],[a_,b_,a_,b_], [70,70,140,140],[70,140,70,140]):
        print(f'{i},{j}, {i_}, {j_}')
        for k in range(8):
            if k!=0:
                perm_upscale[k, j, i] = np.mean(perm_ori [2+(k-1)*1:2+(k)*1, j_, i_])
                poro_upscale[k, j, i] = np.mean(poro_ori [2+(k-1)*1:2+(k)*1, j_, i_])
                rock_upscale[k, j, i] = np.mean(rock_ori [2+(k-1)*1:2+(k)*1, j_, i_])
            else:
                perm_upscale[k, j, i] = np.mean(perm_ori [:2, j_, i_])
                poro_upscale[k, j, i] = np.mean(poro_ori [:2, j_, i_])
                rock_upscale[k, j, i] = np.mean(rock_ori [:2, j_, i_])
    rock_upscale[:2] = 1                
    rock_upscale[2:] [rock_upscale[2:]<2.5] = 2
    rock_upscale[2:] [rock_upscale[2:]>=2.5] = 3
    return poro_upscale, perm_upscale, rock_upscale

poro_upscale, log_perm_upscale, rock_upscale = Upsacling(poro_ori, np.log(perm_ori), rock_ori,nz, ny, nx)
perm_upscale = np.exp(log_perm_upscale)

EN_ROCK = np.array(np.load('Ensemble_v3.npz')['EN_PORO'],dtype = float).reshape(-1,30,128,128)

tempimage = 'tempimage'
if os.path.isdir(tempimage) == False:
    os.mkdir(tempimage)

from PIL import Image

for i in range(30):        
    # Facies
    #plt.imshow(rock_upscale[i,:,:], vmin = 1, vmax = 3)
    plt.imshow(poro_upscale[i,:,:], vmin = 0, vmax = 0.4)
    plt.xlabel('Easting'); plt.ylabel('Northing')   
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)    
    plt.savefig(f'{tempimage}/EnPORO_{i}.png')
    plt.close()

for j in range(29):    
    for i in range(30):        
        # Facies
        # plt.imshow(EN_ROCK[j, i,:,:], vmin = 1, vmax = 3)
        plt.imshow(EN_ROCK[j, i,:,:],  vmin = 0, vmax = 0.4)
        plt.xlabel('Easting'); plt.ylabel('Northing')   
        plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)    
        plt.savefig(f'{tempimage}/EnPORO_{i+30+30*j}.png')
        plt.close()


ImagesPerRow, numRow = 30, 30
images = [Image.open(f'{tempimage}/EnPORO_{i}.png') for i in range(0,30*30)]

widths, heights = zip(*(i.size for i in images))
max_width = max(widths)
max_heights = max(heights)
total_width = max_width * ImagesPerRow
total_height = max_heights * numRow
new_im = Image.new('RGB', (total_width, total_height),
                   color=(255, 255, 255))
x_offset = 0
y_offset = 0
for i in range(ImagesPerRow*numRow):
    new_im.paste(images[i], (x_offset, y_offset))
    x_offset += max_width
    if (i + 1) % ImagesPerRow == 0:
        x_offset = 0
        y_offset += max_heights
new_im.save(f'{tempimage}/EN_PORO_128x128.png')
