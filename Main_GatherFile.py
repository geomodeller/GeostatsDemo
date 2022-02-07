
import os
import numpy as np

EN_ROCK, EN_PORO, EN_LOGPERM = [],[],[]
for i in range(10):
    data = np.load(f'{i}/Ensemble_Subset_CGAN.npz')
    Por = data['Por']
    LogPerm = data['LogPerm']
    Facies = data['Facies']
    EN_ROCK.append(Facies)
    EN_LOGPERM.append(LogPerm)
    EN_PORO.append(Por)
    print('done')
EN_ROCK = np.array(EN_ROCK, dtype = np.int8).reshape(-1,30,128,128)
EN_PORO = np.array(EN_PORO, dtype = np.half).reshape(-1,30,128,128)
EN_LOGPERM = np.array(EN_LOGPERM, dtype = np.float16).reshape(-1,30,128,128)

np.savez('Ensemble_v3.npz',EN_PORO=EN_PORO,EN_ROCK=EN_ROCK,EN_LOGPERM=EN_LOGPERM)