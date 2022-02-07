import numpy as np
import seaborn as sns
from skimage.transform import resize
import matplotlib.pyplot as plt
import pyvista as pv
import glob, os, sys, h5py
import numpy as np
import time
from pyevtk.hl import gridToVTK
from scipy.interpolate import interp1d
from geostatspy.GSLIB import GSLIB2ndarray_3D, Dataframe2GSLIB
from geostatspy import geostats
import pandas as pd
from scipy.stats import truncnorm         # This is for truncated Gaussian distribution
from scipy.stats import norm, rankdata
from scipy import stats         # This is for truncated Gaussian distribution

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

def plot_k_p_sat_ori(permeability, pressure, saturation, nz=7, 
                 wel_loc=[[71,71,141,141],[71,141,141,71]], figsize_ver = 10
                 ):
                 
  color_map = plt.cm.get_cmap('Reds')
  reversed_color_map = color_map.reversed()
  fig = plt.figure(figsize = (20, 2.5))
  for i in range(nz):
      plt.subplot(3, nz, i+1)
      plt.imshow(np.log(permeability[:, :, i]).T, origin='lower', cmap='jet',
                 vmin=-2, vmax=8)

      if i == nz -1:
          plt.colorbar(pad=0.03, fraction=0.045)      
          plt.scatter(wel_loc[0],wel_loc[1], s=1, facecolors='none', edgecolors='k')
      plt.axis('off')
      plt.subplot(3, nz, i+1+nz)
      plt.imshow(saturation[:, :, i].T, origin='lower', cmap=reversed_color_map, vmin=0, vmax=1)

      if i == nz -1:
          plt.colorbar(pad=0.03, fraction=0.045)      
      plt.scatter(wel_loc[0],wel_loc[1], s=1, facecolors='none', edgecolors='k')
      plt.axis('off')
      plt.subplot(3, nz, i+1+nz*2)
      plt.imshow(pressure[:, :, i].T, origin='lower', cmap='jet',
                 vmin=1800, vmax=2300)
      if i == nz -1:
          plt.colorbar(pad=0.03, fraction=0.045)
      plt.axis('off')
  return fig
def plot_k_p_sat(permeability, pressure, saturation, nz=7, 
                 wel_loc=[[71,71,141,141],[71,141,141,71]],
                 ):

  color_map = plt.cm.get_cmap('Reds')
  reversed_color_map = color_map.reversed()
  fig = plt.figure(figsize = (20, 10))
  for i in range(nz):
      plt.subplot(3, nz, i+1)
      plt.imshow(np.log(permeability[:, :, i]).T, origin='lower', cmap='jet',
                 vmin=-2, vmax=8)

      if i == nz -1:
          plt.colorbar(pad=0.03, fraction=0.045)      
          plt.scatter(wel_loc[0],wel_loc[1], s=10, facecolors='none', edgecolors='k')
      plt.axis('off')
      plt.subplot(3, nz, i+1+nz)
      plt.imshow(saturation[:, :, i].T, origin='lower', cmap=reversed_color_map, vmin=0, vmax=1)

      if i == nz -1:
          plt.colorbar(pad=0.03, fraction=0.045)      
      plt.scatter(wel_loc[0],wel_loc[1], s=10, facecolors='none', edgecolors='k')
      plt.axis('off')
      plt.subplot(3, nz, i+1+nz*2)
      plt.imshow(pressure[:, :, i].T, origin='lower', cmap='jet',
                 vmin=1800, vmax=2300)
      if i == nz -1:
          plt.colorbar(pad=0.03, fraction=0.045)
      plt.axis('off')
  return fig

class CMG_Parser():
    def __init__(self, grid_size, grid_dim, top_depth = 0, ft2m = 1):
        self.size = grid_size
        self.X = np.arange(0, grid_dim[0] * (grid_size[0] + 1), grid_dim[0])
        self.Y = np.arange(0, grid_dim[1] * (grid_size[1] + 1), grid_dim[1])
        self.Z = np.arange(top_depth, top_depth + (grid_dim[2] * (grid_size[2] + 1)), grid_dim[2])
            
        self.x = np.zeros((grid_size[0] + 1, grid_size[1] + 1, grid_size[2] + 1))
        self.y = np.zeros((grid_size[0] + 1, grid_size[1] + 1, grid_size[2] + 1))
        self.z = np.zeros((grid_size[0] + 1, grid_size[1] + 1, grid_size[2] + 1))
        
        for k in range(grid_size[2] + 1):
            for j in range(grid_size[1] + 1):
                for i in range(grid_size[0] + 1):
                    self.x[i,j,k] = self.X[i] * ft2m
                    self.y[i,j,k] = self.Y[j] * ft2m
                    self.z[i,j,k] = self.Z[k] * ft2m
                    
        self.attr_names = []
        self.timeSeriesDataList = {}

    def __getitem__(self, key):
        return getattr(self, key)
     
    def writeVTK(self):    
        print("\nWriting VTK file...")  
        if not os.path.exists('VTK'):
            os.makedirs('VTK')
        for t in self.timeSeriesDataList[self.attr_names[0]].keys():          
            pathName = os.getcwd() + "/VTK/" + self.filename_toRead.split('/')[1].split('.')[0] + "_" + t
            data = {}
            for attr_name in self.attr_names:
                data[attr_name] = np.array(self.timeSeriesDataList[attr_name][t]).reshape( (self.size[0], self.size[1], self.size[2]), order="F")    
            gridToVTK(pathName, self.x, self.y, self.z, cellData = data)   
    
    def writeHDF5(self):    
        print("\nWriting HDF5 file...")  
#        if not os.path.exists('HDF5'):
#            os.makedirs('HDF5')
        pathName = os.getcwd() + "/" + self.filename_toRead.split('.')[0] + '.h5'
        hf = h5py.File(pathName, 'w')
        hf.create_dataset('x_coord', data=self.x, compression="gzip", compression_opts=9)
        hf.create_dataset('y_coord', data=self.y, compression="gzip", compression_opts=9)
        hf.create_dataset('z_coord', data=self.z, compression="gzip", compression_opts=9)
        for t in self.timeSeriesDataList[self.attr_names[0]].keys():   
            for attr_name in self.attr_names:
                hf.create_dataset(attr_name + '_' + t, data = np.array(self.timeSeriesDataList[attr_name][t]).reshape( (self.size[0], self.size[1], self.size[2]), order="F"), \
                                  compression="gzip", compression_opts=9)
        hf.close()

    def writeGEOSTable(self):    
        Init_HydroPressure = [1800.9, 1805.4, 1809.8, 1814.3, 1818.8, 1823.2, 1827.7, 1832.1, 1836.6, 1841.1, 1845.5, 1850, 1854.5, 1858.9, 1863.4, 1867.8, 1872.3, 1876.8, 1881.2, 1885.7, 1890.1, 1894.6, 1899, 1903.5, 1908, 1912.4, 1916.9, 1921.3, 1925.8, 1930.3]
        psi2pa = 6894.76
        print("\nWriting GEOS Table...")  
        if not os.path.exists('PressureTable'):
            os.makedirs('PressureTable')
        for t in self.timeSeriesDataList[self.attr_names[0]].keys():          
            pathName = os.getcwd() + "/PressureTable/" + self.filename_toRead.split('/')[1].split('.')[0] + "_" + t + ".Table"   
#            print(pathName)
            for attr_name in self.attr_names:
                num = 0
                with open(pathName, 'w' ) as outputfile:
                    outputfile.write(str(self.size[0]) + " " + str(self.size[1]) + " " + str(self.size[2]) + '\n')
                    for z in range(0, self.size[2]):
                        for col in range(0, self.size[1]):
                            for row in range(0, self.size[0]):
                                pressureChange = self.timeSeriesDataList[attr_name][t][num] - Init_HydroPressure[z]
                                pressureChange = max(pressureChange, 0)
                                outputfile.write(str( pressureChange * psi2pa ) + '\n')
                                num += 1     

    def writeModuli(self):                                
        psi2pa = 6894.76
        C_phi = 3.85e-6
        nv = 0.2
#        K_Rock = [9.5238095e9, 5.56e9, 1.0e10, 5.56e9, 1.0e10]
#        G_Rock = [8.6956521e9, 4.167e9, 6.6e9, 4.167e9, 6.6e9]
        K_Rock = [952380950.24, 5555555556, 10000000000, 5555555556, 10000000000]
        G_Rock = [869565210.74, 4166666667, 6600000000, 4166666667, 6600000000]
        Z_num = 45        
        print("\nWriting Properties Table...")  
        if not os.path.exists('PropertiesTable'):
            os.makedirs('PropertiesTable')
        for t in self.timeSeriesDataList[self.attr_names[0]].keys(): 
            pathNameK = os.getcwd() + "/PropertiesTable/" + self.filename_toRead.split('/')[1].split('.')[0] + "_K.Table"  
            pathNameG = os.getcwd() + "/PropertiesTable/" + self.filename_toRead.split('/')[1].split('.')[0] + "_G.Table" 
            for attr_name in self.attr_names:
                num = 0
                num_temp = 0
                with open(pathNameK, 'w' ) as outputfileK:
                    outputfileG = open(pathNameG, 'w' )
                    outputfileK.write(str(self.size[0]) + " " + str(self.size[1]) + " " + str(Z_num) + '\n')
                    outputfileG.write(str(self.size[0]) + " " + str(self.size[1]) + " " + str(Z_num) + '\n')  
                    for z in range(Z_num):
                        for col in range(0, self.size[1]):
                            for row in range(0, self.size[0]):
                                if z < 2:
                                    outputfileK.write(str( K_Rock[0] ) + '\n')
                                    outputfileG.write(str( G_Rock[0] ) + '\n')                                    
                                elif z < 4:
                                    outputfileK.write(str( K_Rock[1] ) + '\n')
                                    outputfileG.write(str( G_Rock[1] ) + '\n')                                      
                                elif z < 6:
                                    outputfileK.write(str( K_Rock[2] ) + '\n')
                                    outputfileG.write(str( G_Rock[2] ) + '\n')                                     
                                elif z < 8:
                                    outputfileK.write(str( K_Rock[3] ) + '\n')
                                    outputfileG.write(str( G_Rock[3] ) + '\n')                                     
                                elif z < 10:
                                    outputfileK.write(str( K_Rock[4] ) + '\n')
                                    outputfileG.write(str( G_Rock[4] ) + '\n')                                   
                                elif z >= 41 and z < 43: 
                                    outputfileK.write(str( 9.89e9 ) + '\n')
                                    outputfileG.write(str( 6.92e9 ) + '\n')                                    
                                elif z >= 43:
                                    outputfileK.write(str( 2.105e10 ) + '\n')
                                    outputfileG.write(str( 1.205e10 ) + '\n')    
                                elif z == 10:
                                    if (self.timeSeriesDataList[attr_name][t][num_temp] <= 0 and num_temp > 0):
                                        self.timeSeriesDataList[attr_name][t][num_temp] = self.timeSeriesDataList[attr_name][t][num_temp-1]                                    
                                    KMod = (1 - self.timeSeriesDataList[attr_name][t][num_temp]) / self.timeSeriesDataList[attr_name][t][num_temp] / C_phi * psi2pa
                                    KMod = min(max(KMod, 16e9), 34e9)
                                    GMod = 3 * KMod * (1 - 2 * nv) / 2 / (1 + nv)
                                    outputfileK.write(str( KMod ) + '\n')
                                    outputfileG.write(str( GMod ) + '\n')
                                    num_temp += 1  
                                else:
                                    if (self.timeSeriesDataList[attr_name][t][num] <= 0 and num > 0):
                                        self.timeSeriesDataList[attr_name][t][num] = self.timeSeriesDataList[attr_name][t][num-1]                                    
                                    KMod = (1 - self.timeSeriesDataList[attr_name][t][num]) / self.timeSeriesDataList[attr_name][t][num] / C_phi * psi2pa
                                    KMod = min(max(KMod, 16e9), 34e9)
                                    GMod = 3 * KMod * (1 - 2 * nv) / 2 / (1 + nv)
                                    outputfileK.write(str( KMod ) + '\n')
                                    outputfileG.write(str( GMod ) + '\n')
                                    num += 1            
    # copied from ReGrid     
    def buildConstLayer(self, val):
        jKeys = np.arange(self.size[1])
        kLayer = dict((el, []) for el in jKeys)
        for j in range(self.size[1]):
            iRow = []
            for i in range(self.size[0]):
                iRow.append(val)
            kLayer[j] = iRow
        return kLayer
    
    def exportProperty(self, filename_toRead, attr_names, maxTimeStep = 10000000, minTime = 0, maxTime = 1e20):    
        self.filename_toRead = filename_toRead
        logfile = open(os.path.splitext(filename_toRead)[0] + '.LOG', 'w') 
        varIndex = 0
        for attr_name in attr_names:
            print("\nReading ", attr_name, file = logfile)
            varIndex += 1
            attr_name = attr_name.replace(" ", "").strip()
            self.attr_names.append(attr_name)
            setattr(self, attr_name, {})
            # self[attr_name] = {}
            layers = {}
            propIndices = []
            I, J, K = (None,) * 3
            timeDict = []
            t = '0'
            readFlag = False
            timeFlag = False
            numStep = 0
    
            with open(filename_toRead, "r") as fp:
                numline = 0
                for line in fp:
                    numline += 1
                    item = line.split()
                    if len(item) > 0:
                        if item[0] == 'Time' and len(item) > 3:
                            if item[3] == '**********************************************************************':
                                t = item[2].split('.')[0]
                                if int(t) > maxTime:                 
                                    break
                                elif int(t) < minTime:
                                    continue
                                else:
                                    timeFlag = True
                            
                        attr = line.replace(" ", "").strip()
                        if attr == attr_name and timeFlag == True:
                            readFlag = True
                            layers = {}
                            I, J, K = None, None, '1'
                            if numStep + 1 > maxTimeStep:    
                                break
                            numStep += 1
                            timeDict.append(int(t))
                            print("@ Time = ", t, ' ... ', file = logfile) 
                            
                        # Modified from ReGrid     
                        if readFlag:
                            if item[0] == 'All':
                                kKeys = np.arange(self.size[2])
                                layers = dict((el, {}) for el in kKeys)
                                for k in range(self.size[2]):
                                    kLayer = self.buildConstLayer(item[3])
                                    layers[k] = kLayer
                                print("All values are ", item[3], file = logfile)  
                                self[attr_name][t] = layers
                                readFlag = False
                                timeFlag = False
                                continue
                            
                            if item[0] == 'Plane':
                                K = item[3]
                                if len(item) > 4:
                                    if item[4] == 'All':
                                        kLayer = self.buildConstLayer(item[7])
                                        layers[K] = kLayer      
                                        print("Plane ", K, " : buildConstLayer, val = ", item[7], file = logfile)                            
                                        if K == str(self.size[2]):
                                            self[attr_name][t] = layers
                                            layers = {}
                                            readFlag = False
                                            timeFlag = False
                                            continue
                                else:
                                    I = None
                                    layers[K] = {}
    
                            if item[0] == 'I':
                                if K == '1' and I is None:
                                    layers[K] = {}
                                J = None
                                
                                if item[1] == '=':
                                    I = item[2:]
                                else:
                                    I = []
                                    I.append(item[1][1:])
                                    I += item[2:]  
                                    
                                propIndices = []
                                prevDigit = False
                                for i in range(len(line)):
                                    if line[i].isdigit():
                                        if not prevDigit:
                                            propIndices.append(i)
                                            prevDigit = True
                                    else:
                                        prevDigit = False
                                if len(propIndices) != len(I):
                                    print ("len(propIndices) != len(I) : \n", line)
                                    sys.exit()                                            
                                    
                            if item[0][:2] == 'J=':
                                if len(item[0]) == 2:
                                    JIndex = item[1]
                                    J = item[2:]
                                else:
                                    JIndex = item[0][2:]
                                    J = item[1:]      
        
#                                if K == '15' and JIndex == '100' and I[0] == '141' :                             
#                                    print(I)                            
#                                    print(line)
#                                    sys.exit()
                                    
                                if len(J) != len(I):
                                    print('Missing value is found at plane K =', K, " \n", line, file = logfile)                                  
                                    
                                if JIndex not in layers[K].keys():
                                    layers[K][JIndex] = []
                                            
                                skipItem = []
                                for i in range(len(propIndices)):
                                    if (not line[propIndices[i]].isdigit()) and line[propIndices[i]] != '.':
                                        skipItem.append(i)
                                        
                                numSkips = 0
                                for i in range(len(I)):
                                    if i in skipItem:
                                        if layers[K][JIndex] == []:
                                            if JIndex != '1':
                                                layers[K][JIndex].append(layers[K][str(int(JIndex)-1)][i])
                                            else:
                                                layers[K][JIndex].append('-1')                                  
                                        else:
                                            layers[K][JIndex].append(layers[K][JIndex][-1])
                                                
                                        if (i - numSkips) == len(J) or J[i - numSkips] != 'z':
                                            numSkips += 1                                            
                                    else:
                                        if J[i - numSkips] == 'z':
                                            print ("Error: This line has z values: @ plane ", K, " \n", line)
                                            print (propIndices)
                                            print (skipItem)
                                            print (numSkips)
                                            print (i)
                                            print (I)
                                            print (J)
                                            sys.exit()                                    
                                        layers[K][JIndex].append(J[i - numSkips])
                                                                                                
                            if I and J:
                                if int(I[-1]) == self.size[0] and int(JIndex) == self.size[1] and int(K) == self.size[2]:
                                    if readFlag:
                                        self[attr_name][t] = layers
                                        layers = {}
                                        readFlag = False
                                        timeFlag = False
                              
                timeSeriesData = {}
                for t in self[attr_name].keys():
                    timeSeriesData[t] = []
                    for k in self[attr_name][t].keys():
                        for j in self[attr_name][t][k].keys():
                            for val in self[attr_name][t][k][j]:
                                timeSeriesData[t].append(float(val))
            
                self.timeSeriesDataList[attr_name] = timeSeriesData
                
        print('\nTotal number of output time step is ', numStep, ' :', file = logfile)
        print(timeDict, file = logfile)                

"""
attributesToRead = ['Pressure  (psia)', 'Gas Saturation']#, 'Temperature (degF)', 'Current Porosity']
f2m = 0.3048  # ft to m
grid_size = [64,64,8]
grid_dim = [1726.5, 1726.5, 40]
CMGParser = CMG_Parser(grid_size, grid_dim, top_depth, f2m)
"""

def numpy2pyvista(array, nz = 30, aspect = 5):
    # Create the spatial reference
    grid = pv.UniformGrid()
    # Array conversion
    array = array[::-1,:,:].T
    # Set the grid dimensions: shape + 1 because we want to inject our values on
    # the CELL data
    grid.dimensions = np.array(array.shape) + 1
    # Edit the spatial reference
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (1, 1, aspect)  # These are the cell sizes along each axis
    # Add the data values to the cell data
    grid.cell_arrays["values"] = array.flatten(order="F")  # Flatten the array!
    return grid     

def input_inc(poro, perm, rock):
    
    f = open('porosity.inc','w')
    f.writelines('POR ALL \n')
    for i in range(poro.flatten().shape[0]):
        f.writelines(f'{np.round(poro.flatten()[i],4)}\n')
    f.close()
    
    f = open('permeability.inc','w')
    f.writelines('PERMI ALL \n')
    for i in range(perm.flatten().shape[0]):
        f.writelines(f'{perm.flatten()[i]}\n')
    f.close()
    
    f = open('rocktype.inc','w')
    f.writelines('RTYPE ALL  \n')
    for i in range(rock.flatten().shape[0]):
        f.writelines(f'{int(rock.flatten()[i])}\n')
    f.close()

def Upsacling(poro_ori, perm_ori, rock_ori, nz = 8, ny = 64, nx = 64):
    poro_upscale = np.zeros((nz, ny, nx))
    perm_upscale = np.zeros((nz, ny, nx))
    rock_upscale = np.zeros((nz, ny, nx))
    Poro_cap = poro_ori[:2]
    Poro_res = poro_ori[2:]
    Perm_cap = perm_ori[:2]
    Perm_res = perm_ori[2:]
    poro_upscale[0] = resize(Poro_cap, (1,ny, nx), anti_aliasing=False)
    poro_upscale[1:] = resize(Poro_res, (7,ny, nx), anti_aliasing=False)
    perm_upscale[0] = resize(Perm_cap, (1,ny, nx), anti_aliasing=False)
    perm_upscale[1:] = resize(Perm_res, (7,ny, nx), anti_aliasing=False)
    rock_ori = np.array(rock_ori, dtype = float)
    rock_upscale[0] = 1
    rock_upscale[1:] = resize(rock_ori[2:], (7,ny, nx), anti_aliasing=True)
    rock_upscale[1:] [rock_upscale[1:]<2.5] = 2
    rock_upscale[1:] [rock_upscale[1:]>=2.5] = 3
    # Assign hard data
    a_, b_ = int((nx-1)/3),int(2*(nx-1)/3)
    for i, j, i_, j_ in zip([a_,a_,b_,b_],[a_,b_,a_,b_], [70,70,140,140],[70,140,70,140]):
        print(f'{i},{j}, {i_}, {j_}')
        for k in range(8):
            if k!=0:
                perm_upscale[k, j, i] = np.mean(perm_ori [2+(k-1)*4:2+(k)*4, j_, i_])
                poro_upscale[k, j, i] = np.mean(poro_ori [2+(k-1)*4:2+(k)*4, j_, i_])
                rock_upscale[k, j, i] = np.mean(rock_ori [2+(k-1)*4:2+(k)*4, j_, i_])
            else:
                perm_upscale[k, j, i] = np.mean(perm_ori [:2, j_, i_])
                poro_upscale[k, j, i] = np.mean(poro_ori [:2, j_, i_])
                rock_upscale[k, j, i] = np.mean(rock_ori [:2, j_, i_])
    rock_upscale[1:] [rock_upscale[1:]<2.5] = 2
    rock_upscale[1:] [rock_upscale[1:]>=2.5] = 3
    return poro_upscale, perm_upscale, rock_upscale


def Upsacling_Range_preserve(poro_ori, perm_ori, rock_ori, nz = 8, ny = 64, nx = 64):
    poro_upscale = np.zeros((nz, ny, nx))
    perm_upscale = np.zeros((nz, ny, nx))
    rock_upscale = np.zeros((nz, ny, nx))
    Poro_cap = poro_ori[:2]
    Poro_res = poro_ori[2:]
    Perm_cap = perm_ori[:2]
    Perm_res = perm_ori[2:]
    poro_upscale[0] = resize(Poro_cap, (1,ny, nx), anti_aliasing=False , preserve_range = True)
    for i in range(7):
        poro_upscale[i+1] = resize(Poro_res[4*i:4*(i+1)], (1,ny, nx), anti_aliasing=False, preserve_range = True)
    perm_upscale[0] = resize(Perm_cap, (1,ny, nx), anti_aliasing=False , preserve_range = True)
    for i in range(7):
        perm_upscale[i+1] = resize(Perm_res[4*i:4*(i+1)], (1,ny, nx), anti_aliasing=False, preserve_range = True)
        
    rock_ori = np.array(rock_ori, dtype = float)
    rock_upscale[0] = 1
    rock_upscale[1:] = resize(rock_ori[2:], (7,ny, nx), anti_aliasing=True)
    rock_upscale[1:] [rock_upscale[1:]<2.5] = 2
    rock_upscale[1:] [rock_upscale[1:]>=2.5] = 3
    # Assign hard data
    a_, b_ = int((nx-1)/3),int(2*(nx-1)/3)
    for i, j, i_, j_ in zip([a_,a_,b_,b_],[a_,b_,a_,b_], [70,70,140,140],[70,140,70,140]):
        print(f'{i},{j}, {i_}, {j_}')
        for k in range(8):
            if k!=0:
                perm_upscale[k, j, i] = np.mean(perm_ori [2+(k-1)*4:2+(k)*4, j_, i_])
                poro_upscale[k, j, i] = np.mean(poro_ori [2+(k-1)*4:2+(k)*4, j_, i_])
                rock_upscale[k, j, i] = np.mean(rock_ori [2+(k-1)*4:2+(k)*4, j_, i_])
            else:
                perm_upscale[k, j, i] = np.mean(perm_ori [:2, j_, i_])
                poro_upscale[k, j, i] = np.mean(poro_ori [:2, j_, i_])
                rock_upscale[k, j, i] = np.mean(rock_ori [:2, j_, i_])
    rock_upscale[1:] [rock_upscale[1:]<2.5] = 2
    rock_upscale[1:] [rock_upscale[1:]>=2.5] = 3
    return poro_upscale, perm_upscale, rock_upscale



def CDF_mapping(Original, Projected):
	
    Original = Original.flatten() + np.random.normal(scale = 0.00001, size = Original.flatten().shape[0])
    Projected = Projected.flatten()
    
    Original_sort=np.sort(Original); 
    Original_idx = np.zeros(Original.shape[0])
    
    for ii in range(0,Original.shape[0]):
        Original_idx[np.array( Original==Original_sort[ii], dtype=bool)] = ii

    Projected_sort=np.sort(Projected)
    Original_CDF=(Original_idx+1)/float(Original_idx.max()+2)

    Projected_CDF=(np.arange(len(Projected_sort))+1)/float(len(Projected_sort)+2)
    
    f = interp1d(Projected_CDF.T,Projected_sort.T) # CDF mapping via linear interpolation
	
    Original_CDF[Original_CDF<=Projected_CDF.min()] = Projected_CDF.min()
    Original_CDF[Original_CDF>=Projected_CDF.max()] = Projected_CDF.max()
    
    Temp = f(Original_CDF)  
    
    Projected_result =Temp
    return np.array(Projected_result)

def Gaussian_Smoothing_(Original):
    shape = Original.shape
    Original = Original.flatten() + np.random.normal(scale = 0.00001, size = Original.flatten().shape[0])
    Original_CDF = (stats.rankdata(Original)-0.5)/Original.shape[0]
    Projected_sort = norm.ppf(np.linspace(0.00004,0.99996,1000000))

    Projected_CDF=(np.arange(len(Projected_sort))+1)/float(len(Projected_sort)+2)
    f = interp1d(Projected_CDF.T,Projected_sort.T) # CDF mapping via linear interpolation

    Original_CDF[Original_CDF<=Projected_CDF.min()] = Projected_CDF.min()
    Original_CDF[Original_CDF>=Projected_CDF.max()] = Projected_CDF.max()

    Temp = f(Original_CDF)  

    Projected_result =Temp
    return np.array(Projected_result).reshape(shape)


def Gaussian_Smoothing(Original, min_, max_):
    shape = Original.shape
    Original = Original.flatten() + np.random.normal(scale = 0.00001, size = Original.flatten().shape[0])
    Original_CDF = (rankdata(Original)-0.5)/Original.shape[0]
    Projected_sort = truncnorm.ppf(np.linspace(0,1,1000000),min_,max_)

    Projected_CDF=(np.arange(len(Projected_sort))+1)/float(len(Projected_sort)+2)
    f = interp1d(Projected_CDF.T,Projected_sort.T) # CDF mapping via linear interpolation

    Original_CDF[Original_CDF<=Projected_CDF.min()] = Projected_CDF.min()
    Original_CDF[Original_CDF>=Projected_CDF.max()] = Projected_CDF.max()

    Temp = f(Original_CDF)  

    Projected_result =Temp
    return np.array(Projected_result).reshape(shape)


def mapping2Gaussian(Original, mu, std, a, b):
	
    Original = Original.flatten()
    
    Original_sort=np.sort(Original); 
    Original_idx = np.zeros(Original.shape[0])
    
    for ii in range(0,Original.shape[0]):
        Original_idx[np.array( Original==Original_sort[ii], dtype=bool)] = ii

    Original_CDF=(Original_idx+1)/float(len(Original_idx)+2)

    Projected_result = truncnorm.ppf(Original_CDF,a,b) * std + mu
    return np.array(Projected_result)

# %% variogram model of porosity residuals
def sgsim_3d(nreal, df_, xcol, ycol, zcol, vcol, nx_cells, ny_cells, nz, hsiz, vsiz, hmn_max,
             hmn_med, zmn_ver, seed, var, output_file):
    """Sequential Gaussian simulation, 2D wrapper for sgsim from GSLIB (.exe
    must be available in PATH or working directory).


    """
    x = df_[xcol]
    y = df_[ycol]
    z = df_[zcol]
    v = df_[vcol]
    var_min = v.values.min()
    var_max = v.values.max()
    df_temp = pd.DataFrame({"X": x, "Y": y, "Z": z, "Var": v})
    Dataframe2GSLIB("data_temp.dat", df_temp)

    nug = var["nug"]
    nst = var["nst"]
    it1 = var["it1"]
    cc1 = var["cc1"]
    azi1 = var["azi1"]
    dip1 = var["dip1"]
    hmax1 = var["hmax1"]
    hmed1 = var["hmed1"]
    hmin1 = var["hmin1"]
    it2 = var["it2"]
    cc2 = var["cc2"]
    azi2 = var["azi2"]
    dip2 = var["dip2"]
    hmax2 = var["hmax2"]
    hmed2 = var["hmed2"]
    hmin2 = var["hmin2"]
    max_range = max(hmax1, hmax2)
    max_range_v = 1
    hctab = int(max_range / hsiz) * 2 + 1

    with open("sgsim.par", "w") as f:
        f.write("              Parameters for SGSIM                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("data_temp.dat                 -file with data                              \n")
        f.write("1  2  3  4  0  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n")
        f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
        f.write("1                             -transform the data (0=no, 1=yes)            \n")
        f.write("none.trn                      -  file for output trans table               \n")
        f.write("0                             -  consider ref. dist (0=no, 1=yes)          \n")
        f.write("none.dat                      -  file with ref. dist distribution          \n")
        f.write("1  0                          -  columns for vr and wt                     \n")
        f.write(str(var_min) + " " + str(var_max) + "   zmin,zmax(tail extrapolation)       \n")
        f.write("1   " + str(var_min) + "      -  lower tail option, parameter              \n")
        f.write("1   " + str(var_max) + "      -  upper tail option, parameter              \n")
        f.write("0                             -debugging level: 0,1,2,3                    \n")
        f.write("nonw.dbg                      -file for debugging output                   \n")
        f.write(str(output_file) + "           -file for simulation output                  \n")
        f.write(str(nreal) + "                 -number of realizations to generate          \n")
        f.write(str(nx_cells) + " " + str(hmn_max) + " " + str(hsiz) + "                          \n")
        f.write(str(ny_cells) + " " + str(hmn_med) + " " + str(hsiz) + "                          \n")
        f.write(str(nz) + " " + str(zmn_ver) + " " + str(vsiz) + "                          \n")
        f.write(str(seed) + "                  -random number seed                          \n")
        f.write("0     8                       -min and max original data for sim           \n")
        f.write("10                            -number of simulated nodes to use            \n")
        f.write("1                             -assign data to nodes (0=no, 1=yes)          \n")
        f.write("1     3                       -multiple grid search (0=no, 1=yes),num      \n")
        f.write("0                             -maximum data per octant (0=not used)        \n")
        f.write(
            str(max_range) + " " + str(max_range) + " " + str(max_range_v) + " -maximum search  (hmax,hmin,vert) \n")
        f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
        f.write(str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n")
        f.write("1     0.60   1.0              - ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n")
        f.write("none.dat                      -  file with LVM, EXDR, or COLC variable     \n")
        f.write("4                             -  column for secondary variable             \n")
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + "  " + str(dip1) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(
            " " + str(hmax1) + "    " + str(hmed1) + "             " + str(
                hmin1) + "  - a_hmax, a_hmin, a_vert        \n")
        f.write(
            str(it2) + " " + str(cc2) + "   " + str(azi2) + "               " + str(
                dip2) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(" " + str(hmax2) + " " + str(hmed2) + " " + str(hmin2) + " - a_hmax, a_hmin, a_vert        \n")

    os.system("sgsim.exe sgsim.par")
    sim_array = GSLIB2ndarray_3D(output_file, 0, nreal, nx_cells, ny_cells, nz)
    return sim_array[0]

# %% variogram model of porosity residuals
def sgsim_3d_linux(nreal, df_, xcol, ycol, zcol, vcol, nx_cells, ny_cells, nz, hsiz, vsiz, hmn_max,
             hmn_med, zmn_ver, seed, var, output_file):
    """Sequential Gaussian simulation, 2D wrapper for sgsim from GSLIB (.exe
    must be available in PATH or working directory).


    """
    x = df_[xcol]
    y = df_[ycol]
    z = df_[zcol]
    v = df_[vcol]
    var_min = v.values.min()
    var_max = v.values.max()
    df_temp = pd.DataFrame({"X": x, "Y": y, "Z": z, "Var": v})
    Dataframe2GSLIB("data_temp.dat", df_temp)

    nug = var["nug"]
    nst = var["nst"]
    it1 = var["it1"]
    cc1 = var["cc1"]
    azi1 = var["azi1"]
    dip1 = var["dip1"]
    hmax1 = var["hmax1"]
    hmed1 = var["hmed1"]
    hmin1 = var["hmin1"]
    it2 = var["it2"]
    cc2 = var["cc2"]
    azi2 = var["azi2"]
    dip2 = var["dip2"]
    hmax2 = var["hmax2"]
    hmed2 = var["hmed2"]
    hmin2 = var["hmin2"]
    max_range = max(hmax1, hmax2)
    max_range_v = 1
    hctab = int(max_range / hsiz) * 2 + 1

    with open("sgsim.par", "w") as f:
        f.write("              Parameters for SGSIM                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("data_temp.dat                 -file with data                              \n")
        f.write("1  2  3  4  0  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n")
        f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
        f.write("1                             -transform the data (0=no, 1=yes)            \n")
        f.write("none.trn                      -  file for output trans table               \n")
        f.write("0                             -  consider ref. dist (0=no, 1=yes)          \n")
        f.write("none.dat                      -  file with ref. dist distribution          \n")
        f.write("1  0                          -  columns for vr and wt                     \n")
        f.write(str(var_min) + " " + str(var_max) + "   zmin,zmax(tail extrapolation)       \n")
        f.write("1   " + str(var_min) + "      -  lower tail option, parameter              \n")
        f.write("1   " + str(var_max) + "      -  upper tail option, parameter              \n")
        f.write("0                             -debugging level: 0,1,2,3                    \n")
        f.write("nonw.dbg                      -file for debugging output                   \n")
        f.write(str(output_file) + "           -file for simulation output                  \n")
        f.write(str(nreal) + "                 -number of realizations to generate          \n")
        f.write(str(nx_cells) + " " + str(hmn_max) + " " + str(hsiz) + "                          \n")
        f.write(str(ny_cells) + " " + str(hmn_med) + " " + str(hsiz) + "                          \n")
        f.write(str(nz) + " " + str(zmn_ver) + " " + str(vsiz) + "                          \n")
        f.write(str(seed) + "                  -random number seed                          \n")
        f.write("0     8                       -min and max original data for sim           \n")
        f.write("10                            -number of simulated nodes to use            \n")
        f.write("1                             -assign data to nodes (0=no, 1=yes)          \n")
        f.write("1     3                       -multiple grid search (0=no, 1=yes),num      \n")
        f.write("0                             -maximum data per octant (0=not used)        \n")
        f.write(
            str(max_range) + " " + str(max_range) + " " + str(max_range_v) + " -maximum search  (hmax,hmin,vert) \n")
        f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
        f.write(str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n")
        f.write("1     0.60   1.0              - ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n")
        f.write("none.dat                      -  file with LVM, EXDR, or COLC variable     \n")
        f.write("4                             -  column for secondary variable             \n")
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + "  " + str(dip1) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(
            " " + str(hmax1) + "    " + str(hmed1) + "             " + str(
                hmin1) + "  - a_hmax, a_hmin, a_vert        \n")
        f.write(
            str(it2) + " " + str(cc2) + "   " + str(azi2) + "               " + str(
                dip2) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(" " + str(hmax2) + " " + str(hmed2) + " " + str(hmin2) + " - a_hmax, a_hmin, a_vert        \n")

    os.system("./sgsim sgsim.par")
    sim_array = GSLIB2ndarray_3D(output_file, 0, nreal, nx_cells, ny_cells, nz)
    return sim_array[0]


def create_sgs_model(dataframe, vario_dictionary, Val_name, Num_real, grid_dim = [64,64,7], seed = 77777):
    variogram = geostats.make_variogram_3D(**vario_dictionary)
    sgs_model = {
        "nreal": Num_real,
        "df_": dataframe,
        "xcol": "X",
        "ycol": "Y",
        "zcol": "Z",
        "vcol": Val_name,
        "nx_cells": grid_dim[0],
        "ny_cells": grid_dim[1],
        "nz":  grid_dim[2],
        "hsiz": 1,
        "vsiz": 1,
        "hmn_max": 0,
        "hmn_med": 0,
        "zmn_ver": 1,
        "seed": seed,
        "var": variogram,
        "output_file": "sgsim3d.out"
    }
    return sgs_model


def create_sis_model(dataframe, vario_dictionary, Val_name, Num_real, grid_dim = [64,64,7], seed = 77777):
    variogram = geostats.make_variogram_3D(**vario_dictionary)
    sis_model = {
        "nreal": Num_real,
        "df_": dataframe,
        "xcol": "X",
        "ycol": "Y",
        "zcol": "Z",
        "vcol": Val_name,
        "nx_cells": grid_dim[0],
        "ny_cells": grid_dim[1],
        "nz":  grid_dim[2],
        "hsiz": 1,
        "vsiz": 1,
        "hmn_max": 0,
        "hmn_med": 0,
        "zmn_ver": 1,
        "seed": seed,
        "var": variogram,
        "output_file": "sisim.out"
    }
    return sis_model


# %% variogram model of porosity residuals
def sisim_3d(nreal, df_, xcol, ycol, zcol, vcol, nx_cells, ny_cells, nz, hsiz, vsiz, hmn_max,
             hmn_med, zmn_ver, seed, var, output_file):
    """Sequential Gaussian simulation, 2D wrapper for sgsim from GSLIB (.exe
    must be available in PATH or working directory).


    """
    x = df_[xcol]
    y = df_[ycol]
    z = df_[zcol]
    v = df_[vcol]
    var_min = v.values.min()
    var_max = v.values.max()
    df_temp = pd.DataFrame({"X": x, "Y": y, "Z": z, "Var1": v, "Var2": 1-v})
    Dataframe2GSLIB("data_temp.dat", df_temp)

    nug = var["nug"]
    nst = var["nst"]
    it1 = var["it1"]
    cc1 = var["cc1"]
    azi1 = var["azi1"]
    dip1 = var["dip1"]
    hmax1 = var["hmax1"]
    hmed1 = var["hmed1"]
    hmin1 = var["hmin1"]
    it2 = var["it2"]
    cc2 = var["cc2"]
    azi2 = var["azi2"]
    dip2 = var["dip2"]
    hmax2 = var["hmax2"]
    hmed2 = var["hmed2"]
    hmin2 = var["hmin2"]
    max_range = max(hmax1, hmax2)
    max_range_v = 3
    hctab = int(max_range / hsiz) * 2 + 1

    with open("sisim.par", "w") as f:
        f.write("              Parameters for SISIM                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("0                                                                          \n")
        f.write("2                             -  Number of categories                      \n")
        f.write("0 1                           -  Categories                                \n")
        f.write("0.5 0.5                       -  Global CDF                                \n")
        f.write("data_temp.dat                 -  file with data                            \n")        
        f.write("1   2   3   4                 -   columns for X,Y,Z, and variable          \n")
        f.write("none.dat                      -  file with soft data                       \n")
        f.write("1  2  3  4  5  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n")
        f.write("0                             -   Markov-Bayes simulation (0=no,1=yes)     \n")
        f.write("0.61  0.54                    -      calibration B(z) values               \n")
        f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
        
        f.write("0.0   30.0                    -  minimum and maximum data value            \n")
        f.write("1      0.0                    -   lower tail option and parameter          \n")
        f.write("1      1.0                    -   middle     option and parameter          \n")
        f.write("1     30.0                    -   upper tail option and parameter          \n")
        f.write("none.dat                                             \n")
        f.write("3  0                          -  columns for vr and wt                     \n")
        f.write("0                             -debugging level: 0,1,2,3                    \n")
        f.write("sisim.dbg                     -file for debugging output                   \n")
        f.write("sisim.out                     -file for simulation output                  \n")
        f.write(str(nreal) + "                 -number of realizations to generate          \n")
        f.write(str(nx_cells) + " " + str(hmn_max) + " " + str(hsiz) + "                    \n")
        f.write(str(ny_cells) + " " + str(hmn_med) + " " + str(hsiz) + "                    \n")
        f.write(str(nz) + " " + str(zmn_ver) + " " + str(vsiz) + "                          \n")
        f.write(str(seed) + "                  -random number seed                          \n")
        f.write("20                            -number of simulated nodes to use            \n")
        f.write("20                            -number of simulated nodes to use            \n")
        f.write("3                            -number of simulated nodes to use             \n")
        f.write("1                             -assign data to nodes (0=no, 1=yes)          \n")
        f.write("0     3                       -multiple grid search (0=no, 1=yes),num      \n")
        f.write("0                             -maximum data per octant (0=not used)        \n")
        f.write(
            str(max_range) + " " + str(max_range) + " " + str(max_range_v) + " -maximum search  (hmax,hmin,vert) \n")
        f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
        f.write(str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n")
        f.write("0    2.5                      -0=full IK, 1=median approx. (cutoff)         \n")
        f.write("1                             -0=SK, 1=OK                                   \n")
        # Facies 2 (variogram)
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + "  " + str(dip1) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(
            " " + str(hmax1) + "    " + str(hmed1) + "             " + str(
                hmin1) + "  - a_hmax, a_hmin, a_vert        \n")
        # Facies 3 (variogram)
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + "  " + str(dip1) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(
            " " + str(hmax1) + "    " + str(hmed1) + "             " + str(
                hmin1) + "  - a_hmax, a_hmin, a_vert        \n")
    os.system("sisim.exe sisim.par")
    sim_array = GSLIB2ndarray_3D(output_file, 0, nreal, nx_cells, ny_cells, nz)
    return sim_array[0]

# %% variogram model of porosity residuals
def sisim_3d_linux(nreal, df_, xcol, ycol, zcol, vcol, nx_cells, ny_cells, nz, hsiz, vsiz, hmn_max,
             hmn_med, zmn_ver, seed, var, output_file):
    """Sequential Gaussian simulation, 2D wrapper for sgsim from GSLIB (.exe
    must be available in PATH or working directory).


    """
    x = df_[xcol]
    y = df_[ycol]
    z = df_[zcol]
    v = df_[vcol]
    var_min = v.values.min()
    var_max = v.values.max()
    df_temp = pd.DataFrame({"X": x, "Y": y, "Z": z, "Var1": v, "Var2": 1-v})
    Dataframe2GSLIB("data_temp.dat", df_temp)

    nug = var["nug"]
    nst = var["nst"]
    it1 = var["it1"]
    cc1 = var["cc1"]
    azi1 = var["azi1"]
    dip1 = var["dip1"]
    hmax1 = var["hmax1"]
    hmed1 = var["hmed1"]
    hmin1 = var["hmin1"]
    it2 = var["it2"]
    cc2 = var["cc2"]
    azi2 = var["azi2"]
    dip2 = var["dip2"]
    hmax2 = var["hmax2"]
    hmed2 = var["hmed2"]
    hmin2 = var["hmin2"]
    max_range = max(hmax1, hmax2)
    max_range_v = 3
    hctab = int(max_range / hsiz) * 2 + 1

    with open("sisim.par", "w") as f:
        f.write("              Parameters for SISIM                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("0                                                                          \n")
        f.write("2                             -  Number of categories                      \n")
        f.write("0 1                           -  Categories                                \n")
        f.write("0.5 0.5                       -  Global CDF                                \n")
        f.write("data_temp.dat                 -  file with data                            \n")        
        f.write("1   2   3   4                 -   columns for X,Y,Z, and variable          \n")
        f.write("none.dat                      -  file with soft data                       \n")
        f.write("1  2  3  4  5  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n")
        f.write("0                             -   Markov-Bayes simulation (0=no,1=yes)     \n")
        f.write("0.61  0.54                    -      calibration B(z) values               \n")
        f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
        
        f.write("0.0   30.0                    -  minimum and maximum data value            \n")
        f.write("1      0.0                    -   lower tail option and parameter          \n")
        f.write("1      1.0                    -   middle     option and parameter          \n")
        f.write("1     30.0                    -   upper tail option and parameter          \n")
        f.write("none.dat                                             \n")
        f.write("3  0                          -  columns for vr and wt                     \n")
        f.write("0                             -debugging level: 0,1,2,3                    \n")
        f.write("sisim.dbg                     -file for debugging output                   \n")
        f.write("sisim.out                     -file for simulation output                  \n")
        f.write(str(nreal) + "                 -number of realizations to generate          \n")
        f.write(str(nx_cells) + " " + str(hmn_max) + " " + str(hsiz) + "                    \n")
        f.write(str(ny_cells) + " " + str(hmn_med) + " " + str(hsiz) + "                    \n")
        f.write(str(nz) + " " + str(zmn_ver) + " " + str(vsiz) + "                          \n")
        f.write(str(seed) + "                  -random number seed                          \n")
        f.write("20                            -number of simulated nodes to use            \n")
        f.write("20                            -number of simulated nodes to use            \n")
        f.write("3                            -number of simulated nodes to use             \n")
        f.write("1                             -assign data to nodes (0=no, 1=yes)          \n")
        f.write("0     3                       -multiple grid search (0=no, 1=yes),num      \n")
        f.write("0                             -maximum data per octant (0=not used)        \n")
        f.write(
            str(max_range) + " " + str(max_range) + " " + str(max_range_v) + " -maximum search  (hmax,hmin,vert) \n")
        f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
        f.write(str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n")
        f.write("0    2.5                      -0=full IK, 1=median approx. (cutoff)         \n")
        f.write("1                             -0=SK, 1=OK                                   \n")
        # Facies 2 (variogram)
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + "  " + str(dip1) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(
            " " + str(hmax1) + "    " + str(hmed1) + "             " + str(
                hmin1) + "  - a_hmax, a_hmin, a_vert        \n")
        # Facies 3 (variogram)
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + "  " + str(dip1) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(
            " " + str(hmax1) + "    " + str(hmed1) + "             " + str(
                hmin1) + "  - a_hmax, a_hmin, a_vert        \n")
    os.system("./sisim sisim.par")
    sim_array = GSLIB2ndarray_3D(output_file, 0, nreal, nx_cells, ny_cells, nz)
    return sim_array[0]



def sgs_realizations(sgs_model_dict):
    #for i in range(n_realizations):
    #sgs_model_dict['seed'] = i + 5
    sim = sgsim_3d(**sgs_model_dict)
    #tensor[i, ...] = sim[0, ...]

    return sim



def sis_realizations(sis_model_dict):
    #for i in range(n_realizations):
    #sgs_model_dict['seed'] = i + 5
    sim = sisim_3d(**sis_model_dict)
    #tensor[i, ...] = sim[0, ...]

    return sim



def sgs_realizations_linux(sgs_model_dict):
    #for i in range(n_realizations):
    #sgs_model_dict['seed'] = i + 5
    sim = sgsim_3d_linux(**sgs_model_dict)
    #tensor[i, ...] = sim[0, ...]

    return sim



def sis_realizations_linux(sis_model_dict):
    #for i in range(n_realizations):
    #sgs_model_dict['seed'] = i + 5
    sim = sisim_3d_linux(**sis_model_dict)
    #tensor[i, ...] = sim[0, ...]

    return sim

