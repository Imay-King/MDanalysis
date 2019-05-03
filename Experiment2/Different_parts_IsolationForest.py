#!/usr/bin/python
# -*- coding: UTF-8 -*-
#!/usr/bin/python
# -*- coding: UTF-8 -*-
#polluted  train data
#outlier detection  if the training data contains outliers which are defined as observations that are far from the others
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np
import MDAnalysis as mda
import MDAnalysis.coordinates.DCD
import MDAnalysis.coordinates.DCD as dcd
import MDAnalysis.analysis.rms
import matplotlib.pyplot as plt
def generateData(filename_train,filename_test):
    data=pd.read_csv(filename_train,header=None)
    test_data=pd.read_csv(filename_test,header=None)

    train_data=data.iloc[:,:-2]
    test_data=test_data.iloc[:,:-1]


    #normalize the dataset
    scalar = MinMaxScaler()
    train_data=scalar.fit(train_data).transform(train_data)
    #new_test_data=scalar.fit_transform(test_data)
    scalar_test=StandardScaler()
    test_data=scalar_test.fit_transform(test_data)

    rng = np.random.RandomState(10)
    clf = IsolationForest(n_estimators=200,behaviour='new', max_samples=200,max_features=5,
                          random_state=rng, contamination='auto')
    clf.fit(train_data)  #train data
    pre_label=clf.fit_predict(test_data)
    print(pre_label)
    count=0
    index_numbers=[]
    for index,i in enumerate(pre_label):
        if(i==-1):
            count+=1
            index_numbers.append(index)
    print(index_numbers)
    print('The negative sample is :',count)
    return index_numbers
def plot_graphs(R,R_res,list):

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    rmsd_res = R_res.rmsd.T  # transpose makes it easier for plotting
    time_res = rmsd_res[1][::5]
    rmsd = R.rmsd.T  # transpose makes it easier for plotting
    time = rmsd[1][::5]
    frames = np.arange(len(time))
    figure, ax = plt.subplots(figsize=(15, 15))
    plot_original, = plt.plot(frames, rmsd[2][::5], 'b--', label="Original Frames")
    plot_res, = plt.plot(frames, rmsd_res[2][::5], 'g--', label="Restrained Frames")
    for i in list:
        plt.plot(i, rmsd[2][::5][i], 'o', c='red')
    plt.legend(loc="best")
    plt.xlabel("Frames", font1)
    plt.ylabel(r"RMSD ($\AA$)", font1)
    plt.tick_params(labelsize=16)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print labels
    [label.set_fontname('Times New Roman') for label in labels]
    my_x_ticks = np.arange(0, 501, 100)
    plt.legend(handles=[plot_original, plot_res], prop=font1)
    plt.xticks(my_x_ticks)
    plt.show()

if __name__ == '__main__':
    topology = 'ala13_mini_conv.pdb'
    trajectory = 'ala13_md.dcd'
    universe = mda.Universe(topology, trajectory)
    trajectory_res = 'ala13_md_res_ca.dcd'
    universe_res = mda.Universe(topology, trajectory_res)
    R1 = MDAnalysis.analysis.rms.RMSD(universe,
                                      select='resid 1:4 and not name H*')
    R1.run()
    R2 = MDAnalysis.analysis.rms.RMSD(universe,
                                      select='resid 5:9 and not name H*')
    R2.run()
    R3 = MDAnalysis.analysis.rms.RMSD(universe,
                                      select='resid 10:13 and not name H*')
    R3.run()
    R1_res = MDAnalysis.analysis.rms.RMSD(universe_res,
                                          select='resid 1:4 and not name H*')
    R1_res.run()
    R2_res = MDAnalysis.analysis.rms.RMSD(universe_res,
                                          select='resid 5:9 and not name H*')
    R2_res.run()
    R3_res = MDAnalysis.analysis.rms.RMSD(universe_res,
                                          select='resid 10:13 and not name H*')
    R3_res.run()
    start=time.time()
    list_part1=generateData("train_X_res_ca_1-4.csv","test_X_unstable_1-4.csv")
    list_part2=generateData("train_X_res_ca_5-9.csv", "test_X_unstable_5-9.csv")
    list_part3=generateData("train_X_res_ca_10-13.csv", "test_X_unstable_10-13.csv")
    end=time.time()
    print('run time for program:',end-start)
    #generateData("train_X_res_ca.csv", "test_X_unstable.csv")
    #generateData("train_X_res_ca_shuffle_training.csv",'train_X_res_ca_shuffle_testing.csv')
    plot_graphs(R1,R1_res,list_part1)
    plot_graphs(R2,R2_res,list_part2)
    plot_graphs(R3,R3_res,list_part3)
