import numpy as np
import shutil
from subprocess import call
#import h5py
import multiprocessing
import os
import time

# This is a script to automatically generate random field from R.


##########################################################################################
# MAIN LOOP
##########################################################################################
def create_folder(n):
    for i in range(n):
        if os.path.isdir(f'ParaSubset_{i}') == False:
            os.mkdir(f'ParaSubset_{i}')
        shutil.copy('Numpy_P50_permeability.npy',f'ParaSubset_{i}/Numpy_P50_permeability.npy')
        shutil.copy('Numpy_P50_porosity.npy',f'ParaSubset_{i}/Numpy_P50_porosity.npy')
        shutil.copy('Numpy_P50_RockType.npy',f'ParaSubset_{i}/Numpy_P50_RockType.npy')
        shutil.copy('sgsim',f'ParaSubset_{i}/sgsim')
        shutil.copy('sisim',f'ParaSubset_{i}/sisim')
        shutil.copy('Main_Generate_Ensemble_128x128x30_v3.py',f'ParaSubset_{i}/Main_Generate_Ensemble_128x128x30_v3.py')
        shutil.copy('Sub.py',f'ParaSubset_{i}/Sub.py')
    
def run_Rscript(i, num_per_run):
    os.chdir(f'ParaSubset_{i}')
    import Main_Generate_Ensemble_128x128x30_v3.py
    #call(f'python Main_Generate_Ensemble_128x128x30_v2.py')


def run_simulator_par_all(n, num_per_run):
    num = np.arange(0, n, 1)
    print(multiprocessing.cpu_count())
    processes = []

    for i in range(0, n):
        p = multiprocessing.Process(target=run_Rscript, args=(num[i], num_per_run))
        os.system("taskset -p -c %d %d" % (i % multiprocessing.cpu_count(), os.getpid()))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    # permeability = collect_data(n)


# def collect_data(n):
#     dirName = os.getcwd()
#     datafilename = 'permeability.hdf5'
    
#     for i in range(0, n):
#         np.savez(f'Ensemble_Subset_.npz', Por = SGS_por_total, LogPerm = SGS_logPerm_total, Facies = SIS_total)   
        
#         perm_pool = np.append(perm_pool, perm, axis=0)

#     return perm_pool

if __name__ == '__main__':
    start = time.perf_counter()
    NPar = 36
    num_per_run = 100
    create_folder(NPar)
    run_simulator_par_all(NPar, 1)
    import Main_GatherFile.py
    import Main_GatherVisual.py


