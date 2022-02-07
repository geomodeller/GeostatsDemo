# =============================================================================
# Generate Ensemble with Geostatistics (128x128x30):
# Date: 7/22/2021
# Authors: Pengcheng, Hewei, Su, Honggeun   
# =============================================================================
# This workflow is based on Pengcheng's suggestions
# Key purposes are 1) MAKE ENSEMBLE CONSISTENT TO REFERENCE 
#            WHILE 2) PRESERVING DIVERSE RESPONSES
# Workflow:
# 1) Read Original Model
# 2) Rescale to 128x128x30
# 3) NST Porosity and Permeabilty
# (iterations)
#   4) Extract resampled "hard data" (facies and NST_poro and NST_perm)
#   5) SIS for facies and SGS for porosity and permeability
#   6) Save multiple realizations
# =============================================================================
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.transform import resize
import matplotlib.pyplot as plt
import pyvista as pv
import Sub
import glob
from PIL import Image
from sklearn.preprocessing import PowerTransformer
import geostatspy.GSLIB as GSLIB          # GSLIB utilies, visualization and wrapper
import geostatspy.geostats as geostats    # GSLIB methods convert to Python 
from scipy.interpolate import interp1d
from sys import platform
from skimage.filters import gaussian
import shutil
# =============================================================================
# Load original Model
# =============================================================================
poro_ori = np.load('Numpy_P50_porosity.npy')
perm_ori = np.load('Numpy_P50_permeability.npy')
rock_ori = np.load('Numpy_P50_RockType.npy')
cmap = plt.cm.get_cmap("jet")
# =============================================================================
# define some parameters
# =============================================================================
nz_fine = 30
ny_fine = 211
nx_fine = 211

nz, ny, nx = 30, 128, 128
nz_cap = 2

poro_upscale, log_perm_upscale, rock_upscale = Sub.Upsacling(poro_ori, np.log(perm_ori), rock_ori,nz, ny, nx)
perm_upscale = np.exp(log_perm_upscale)

# =============================================================================
# Collect statistics and NST poro&perm 
# =============================================================================
# 1 caprock 2 shale 3 sand
poro_mean_sh = np.mean(poro_upscale[rock_upscale==2])
poro_std_sh = np.std(poro_upscale[rock_upscale==2])
poro_mean_s = np.mean(poro_upscale[rock_upscale==3])
poro_std_s = np.std(poro_upscale[rock_upscale==3])
print('Porosity (shaly sand): mean and std ' + str(poro_mean_sh) + ' ' + str(poro_std_sh))
print('Porosity (sand): mean and std ' + str(poro_mean_s) + ' ' + str(poro_std_s))
logperm_mean_sh = np.mean(np.log(perm_upscale[rock_upscale==2]))
logperm_std_sh = np.std(np.log(perm_upscale[rock_upscale==2]))
logperm_mean_s = np.mean(np.log(perm_upscale[rock_upscale==3]))
logperm_std_s = np.std(np.log(perm_upscale[rock_upscale==3]))
print('LogPerm (shaly sand): mean and std ' + str(logperm_mean_sh) + ' ' + str(logperm_std_sh))
print('LogPerm (sand): mean and std ' + str(logperm_mean_s) + ' ' + str(logperm_std_s))

## normalize both permeability and porosity for facies 2 and 3
poro_upscale_norm = poro_upscale.copy()
logperm_upscale_norm = np.log(perm_upscale.copy())
poro_upscale_norm[rock_upscale==2] = (poro_upscale[rock_upscale==2] - poro_mean_sh) / poro_std_sh 
poro_upscale_norm[rock_upscale==3] = (poro_upscale[rock_upscale==3] - poro_mean_s) / poro_std_s 
logperm_upscale_norm[rock_upscale==2] = (np.log(perm_upscale[rock_upscale==2]) - logperm_mean_sh) / logperm_std_sh 
logperm_upscale_norm[rock_upscale==3] = (np.log(perm_upscale[rock_upscale==3]) - logperm_mean_s) / logperm_std_s 

# Min and Max for NST Por and Perm for each facies
nst_por_min_SS = poro_upscale_norm[rock_upscale==3].min()
nst_por_min_SHS = poro_upscale_norm[rock_upscale==2].min()
nst_por_max_SS = poro_upscale_norm[rock_upscale==3].max()
nst_por_max_SHS = poro_upscale_norm[rock_upscale==2].max()
nst_lopPerm_min_SS = logperm_upscale_norm[rock_upscale==3].min()
nst_lopPerm_min_SHS = logperm_upscale_norm[rock_upscale==2].min()
nst_lopPerm_max_SS = logperm_upscale_norm[rock_upscale==3].max()
nst_lopPerm_max_SHS = logperm_upscale_norm[rock_upscale==2].max()

# =============================================================================
# Define well locations
# =============================================================================
## well locations
a_, b_ = int((nx-1)/3),int(2*(nx-1)/3)
well_loc_x = [a_, b_]
well_loc_y = [a_, b_]
x, y = np.meshgrid(well_loc_x, well_loc_y)   # <- Injectors
# x_,y_ = np.array([25, 104, 104, 104, 190]), np.array([104, 190, 104, 25, 104])
x_,y_ = np.array([23, 63, 63, 63, 127-23, 5, 5, 127-5,127-5, 23,23,127-23,127-23]), np.array([63, 127-23, 63, 23, 63,5, 127-5, 127-5,5,23,127-23,127-23,23])

## hard data extract
x = np.concatenate([x.flatten(),x_.flatten()]);
y = np.concatenate([y.flatten(),y_.flatten()]);
x = x.flatten()
y = y.flatten()

# =============================================================================
# Load hard data and initial settings for Geostatistics
# =============================================================================
#Load hard data
df = pd.read_csv('Hard_data.csv')
df_all = df.copy()

# Default variogram parameters
vario_sis = { "nug": 0, # Nugger
              "nst": 1, # Num. of structure
              "it1": 1, # Variogram type( 1-> exponential, 2,3-> Gaussian and Spherical)
              "cc1": 1, # Sill
              "azi1": 90, # Azimuth of major direction
              "dip1": 0.0, # No dip
              "hmax1": 60,  # Major range (in grid block unit), it's 50,000 ft
              "hmed1": 30,  # Minor range (in grid block unit), it's 25,000 ft
              "hmin1": 4.5    # Vertical range
}
vario_sgs = { "nug": 0, # Nugger
              "nst": 1, # Num. of structure
              "it1": 1, # Variogram type( 1-> exponential, 2,3-> Gaussian and Spherical)
              "cc1": 1, # Sill
              "azi1": 0.0, # Azimuth of major direction
              "dip1": 0.0, # No dip
              "hmax1": 5, # Major range
              "hmed1": 5,  # Minor range
              "hmin1": 1     # Vertical range
}


# =============================================================================
# Geostatistical part
# =============================================================================
SIS_total = [];
SGS_por_total = [];
SGS_logPerm_total = [];
Vario_para_total =[];
for iteration in range(1):

    ## Randomly select hard data in well locations
    loc_rock, Rock  = [], []
    Rock_Por, Rock_Perm =[], []
    for i in range(x.shape[0]):
        for k in range(2, 30):
            x_dummy, y_dummy = np.random.choice(127), np.random.choice(127)
            loc_ = [x[i],y[i],k]
            Rock.append(rock_upscale[k, y_dummy, x_dummy])
            Rock_Por.append(poro_upscale_norm[k, y_dummy, x_dummy])
            Rock_Perm.append(logperm_upscale_norm[k, y_dummy, x_dummy])
            loc_rock.append(loc_)

    # Pandas DataFrame for hard data: 
    loc_rock = np.array(loc_rock).reshape(-1,3)
    Rock = np.array(Rock).flatten() - 2
    Rock = Rock.reshape(-1,1)
    Rock_Por, Rock_Perm = np.array(Rock_Por).reshape(-1,1), np.array(Rock_Perm).reshape(-1,1), 
    data = np.concatenate([loc_rock, Rock, Rock_Por, Rock_Perm], axis = 1)
    df = pd.DataFrame(data = data, columns = ['X', 'Y', 'Z', 'Rock Facies', 'Por', 'LogPerm'])
    df.head(n = 20)
    df['Rock Facies'] = df['Rock Facies'].astype(int) 
    df.Z -= 1
    df_all = df.copy()
    
    ## Add uncertainty in continuity range / direction
    vario_sis['azi1'] = np.random.uniform(90, 90, 1)[0]
    vario_sis['hmax1'] = np.random.uniform(70, 70, 1)[0]*2
    vario_sis['hmed1'] = np.random.uniform(30, 30, 1)[0]*2
    vario_sis['hmin1'] = np.random.uniform(6, 6, 1)[0]
    vario_sgs['hmax1'] = np.random.uniform(5.5, 6, 1)[0]
    vario_sgs['hmed1'] = vario_sgs['hmax1']
    Vario_para_ = [vario_sis['azi1'], vario_sis['hmax1'],vario_sis['hmed1'],vario_sis['hmin1'],vario_sgs['hmax1'],vario_sgs['hmed1']]
    Vario_para_total.append(Vario_para_)
    
    ## Add measurement noise (5% Gaussian error)        
    df_temp_all = df_all.copy()
    df_temp_all ['Por'] = df_temp_all ['Por'] * (1 + np.random.normal(scale = .05,size = len(df_temp_all)))
    df_temp_all ['LogPerm'] = df_temp_all ['LogPerm'] *  (1 + np.random.normal(scale = 0.05,size = len(df_temp_all)))
    
    ## Soft Thresholding for injector locations - To prevent extreme values in well locations
    df_temp_all[:28*4].Por [df_temp_all[:28*4].Por>2.5] = 2.5
    df_temp_all[:28*4].Por [df_temp_all[:28*4].Por<-2.5] = -2.5
    df_temp_all[:28*4].LogPerm [df_temp_all[:28*4].LogPerm>2.5] = 2.5
    df_temp_all[:28*4].LogPerm [df_temp_all[:28*4].LogPerm<-2.5] = -2.5
    
    ## Draw random seed - should change in each realization
    seed_1 = int(np.random.uniform(0,1)*100000)
    seed_2 = int(np.random.uniform(0,1)*100000)

    # =============================================================================
    #   Indicator Kriging (or SIS) or facise
    # =============================================================================
    Input_sis = Sub.create_sis_model(df, vario_sis, grid_dim =[nx, ny, nz-nz_cap], Val_name = 'Rock Facies', Num_real = 1, seed = seed_1)
    if platform[0] == 'w':
        sis = Sub.sis_realizations(Input_sis)
    else:
        sis = Sub.sis_realizations_linux(Input_sis)
    sis = sis[:,::-1,:,:]         # in num_real, Y, X, and Z without caprock
    sis = np.moveaxis(sis, -1, 1) # Now it's in num_real x dz x dy x dx
    sis_ = np.zeros((sis.shape[0],nz,ny,nx))
    sis_[:,:nz_cap] = rock_upscale[:nz_cap]
    sis_[:,nz_cap:] = sis + 2     # To make 1, 2, and 3
    
    # This is for smoothing the boundaries of facies models
    sis_gaussian = []
    for i in range(30):
        temp = gaussian(sis_[0,i], 3, preserve_range=False)
        temp[temp>2.7]=3
        temp[temp<=2.3]=2
        temp_inter = temp[(temp<2.7)&(temp>2.3)]
        Prob = (temp[(temp<2.7)&(temp>2.3)]-2.3)*2.5
        for i in range(temp_inter.shape[0]):
            prob = Prob[i]
            # prob = prob **3 -1.5*prob**2 +1.5*prob 
            prob = np.array([1-prob,prob])
            temp_inter [i] = np.random.choice(2,size=1, p = prob)+2
        temp[(temp<2.7)&(temp>2.3)] = temp_inter
        sis_gaussian.append(temp)
    sis_gaussian = np.array(sis_gaussian,dtype = float).reshape(sis_.shape)
    sis_gaussian[0,:2] = 1
    sis_ = sis_gaussian
    
    # =============================================================================
    #   Kriging (or SGS) for porosity and permeability
    # =============================================================================
    ## Porosity
    Input_sgs = Sub.create_sgs_model(df_temp_all, vario_sgs, grid_dim =[nx, ny, nz-nz_cap], Val_name = 'Por', Num_real = 1, seed=seed_2)
    if platform[0] == 'w':
        sgs = Sub.sgs_realizations(Input_sgs)
    else:
        sgs = Sub.sgs_realizations_linux(Input_sgs)
    sgs = sgs[:,::-1,:,:] # in X, Y, and Z without caprock
    sgs_por = np.moveaxis(sgs, -1, 1) # Now it's in num_real x dz x dy x dx
    # Save hard data before posterior process
    sgs_por_hard = -100*np.ones(sgs_por.shape)
    sgs_por_hard [sis==0] = sgs_por[sis==0] * poro_std_sh + poro_mean_sh
    sgs_por_hard [sis==1] = sgs_por[sis==1] * poro_std_s + poro_mean_s
    sgs_por[sis_[:,2:]==2] = Sub.Gaussian_Smoothing(sgs_por[sis_[:,2:]==2], nst_por_min_SHS, nst_por_max_SHS)
    sgs_por[sis_[:,2:]==3] = Sub.Gaussian_Smoothing(sgs_por[sis_[:,2:]==3], nst_por_min_SS, nst_por_max_SS)
    sgs_por_ = np.zeros((1,nz,ny,nx))
    sgs_por_ [:,:nz_cap]  = poro_upscale[:nz_cap]
    sgs_por_ [:,nz_cap:] = sgs_por
    
    ## Log Permeability
    Input_sgs = Sub.create_sgs_model(df_temp_all, vario_sgs, grid_dim =[nx, ny, nz-nz_cap], Val_name = 'LogPerm', Num_real = 1, seed=seed_2)
    if platform[0] == 'w':
        sgs = Sub.sgs_realizations(Input_sgs)
    else:
        sgs = Sub.sgs_realizations_linux(Input_sgs)
    sgs = sgs[:,::-1,:,:] # in X, Y, and Z without caprock
    sgs_logPerm = np.moveaxis(sgs, -1, 1) # Now it's in num_real x dz x dy x dx
    # Save hard data before posterior process
    sgs_logPerm_hard = -100*np.ones(sgs_logPerm.shape)
    sgs_logPerm_hard [sis==0] = sgs_logPerm[sis==0] * logperm_std_sh + logperm_mean_sh
    sgs_logPerm_hard [sis==1] = sgs_logPerm[sis==1] * logperm_std_s + logperm_mean_s
    sgs_logPerm[sis_[:,2:]==2] = Sub.Gaussian_Smoothing(sgs_logPerm[sis_[:,2:]==2], nst_lopPerm_min_SHS, nst_lopPerm_max_SHS)
    sgs_logPerm[sis_[:,2:]==3] = Sub.Gaussian_Smoothing(sgs_logPerm[sis_[:,2:]==3], nst_lopPerm_min_SS, nst_lopPerm_max_SS)
    sgs_logPerm_ = np.zeros((1,nz, ny, nx))
    sgs_logPerm_ [:,:nz_cap]  = np.log(perm_upscale[:nz_cap])
    sgs_logPerm_ [:,nz_cap:] = sgs_logPerm

    # =============================================================================
    # Combine two facies models into one 
    # =============================================================================
    sgs_por = np.zeros(sgs_por_.shape)
    sgs_logPerm = np.zeros(sgs_logPerm_.shape)
    # Cap rock - 1
    sgs_por [sis_==1] =  sgs_por_ [sis_==1]
    sgs_logPerm [sis_==1] =  sgs_logPerm_ [sis_==1]
    # Sandstone - 3
    sgs_por [sis_==3] = sgs_por_[sis_==3] * poro_std_s + poro_mean_s
    sgs_logPerm [sis_==3] =  sgs_logPerm_ [sis_==3] * logperm_std_s + logperm_mean_s
    # ShalySand - 2
    sgs_por [sis_==2] = sgs_por_[sis_==2] * poro_std_sh + poro_mean_sh
    sgs_logPerm [sis_==2] =  sgs_logPerm_ [sis_==2] * logperm_std_sh + logperm_mean_sh  
    
    # Assing hard data
    for j in well_loc_x:
        for k in well_loc_y: 
            sgs_por[:,nz_cap:,j,k] =  sgs_por_hard[:,:,j, k] 
            sgs_logPerm[:,nz_cap:,j,k] =  sgs_logPerm_hard[:,:,j,k] 
    
    # another thresholding to remove extreme outliers
    sgs_por[sgs_por<0.001] = 0.001
    sgs_logPerm[sgs_logPerm>9.5] = 9.5
    sgs_logPerm[sgs_logPerm<-7.3] = -7.3
    
    # Save the outcomes
    SIS_total.append(sis_)
    SGS_por_total.append(sgs_por)
    SGS_logPerm_total.append(sgs_logPerm)
    print(f'iteration {iteration+1} out of 250')
        
# Convert list type to np.array
SIS_total = np.array(SIS_total, dtype = np.int8)
SGS_por_total = np.array(SGS_por_total, dtype = np.half)
SGS_logPerm_total = np.array(SGS_logPerm_total, dtype = np.float16)
Vario_para_total = np.array(Vario_para_total)
np.savez('Ensemble_Subset_CGAN.npz', Por = SGS_por_total, LogPerm = SGS_logPerm_total, Facies = SIS_total, Vario_para_total= Vario_para_total)   
