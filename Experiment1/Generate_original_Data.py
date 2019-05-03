#!/usr/bin/python
# -*- coding: UTF-8 -*-'
import MDAnalysis as mda
import MDAnalysis.coordinates.DCD
import MDAnalysis.coordinates.DCD as dcd
import numpy as np
import pandas as pd
topology='ala5_min_conv.pdb'
trajectory='ala5_first_trajectory.dcd'
universe=mda.Universe(topology,trajectory)
f=dcd.DCDFile(trajectory);
data=[]
ts=[]
atom_names = np.tile(np.array(universe.atoms.names),len(universe.trajectory)) # length of trajectory is 29
#print(atom_names)
for time_step in universe.trajectory:
    ts.append(universe.trajectory.time)
    time_point=universe.trajectory.time
    for i in range(len(universe.atoms)):    #there are 53 atoms in total
        pure_data = np.append(universe.atoms.positions[i], round(time_point,4))
        #print(pure_data)
        data.append(pure_data)
#print(ts)
#print(len(ts))
print(data)
#print(position_data)
#data=data.append(atom_name)
#position_data=pd.DataFrame(data,columns=['X_co', 'Y_co', 'Z_co'],index=atom_names,dtype='float32')
position_data=pd.DataFrame(data,columns=['X_co', 'Y_co', 'Z_co','time_point'],dtype='float32')
#write the file
position_data.to_csv('first.csv',index=False,header=False)
print(position_data)







