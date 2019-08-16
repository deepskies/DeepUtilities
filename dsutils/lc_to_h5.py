import os

import pandas as pd
import numpy as np
import h5py
import glob 
import cv2
from astropy.io import fits

# from cosmoNODE.loaders import Big 


class IO:
    def __init__(self, filepath):
        """
        * Loads light curve data
        * Creates the h5 file
        * 

        """
        self.filepath = filepath 
        
       
        os.getcwd()
        self.df = pd.read_csv('../../cosmoNODE/demos/data/training_set.csv') #data filepath
        self.labels = pd.read_csv('../../cosmoNODE/demos/data/training_set_metadata.csv') #labels filepath
 
        # main dataset h5 file
        self.f = h5py.File("lc.hdf5", 'w')
    
    def image_set(self):
        ''' Load labels + Image Data '''
        columns = labels.columns
        labels = []
        for objdict in columns:
            label = labels[objdict]
            label = np.asarray(label)
            labels.append(label)

        image_data = []
        for filename in glob.glob('/*.fits', filepath): 
            im=fits.open(filename)
            img_data = im[0].data
            img_data = img_data.reshape(-1, 101,101)
            image_data.append(img_data)    

    def process(self):
        """
        * Creates the h5 groups 
        * Puts light curve data in them
        * 
        * 
        """
        # df = self.lcs.df
        # meta = self.lcs.meta
        df = self.df #data
        labels = self.labels #labels
        

        
        groups = df.groupby(self.id_col)
        
        X = self.f.create_group('X')
        Y = self.f.create_group('Y')
        
        X = X.create_dataset('LensData.h5', data=image_data)
        Y = Y.create_dataset('LensLabels.h5', data=labels)

        for i, group in enumerate(groups):
            group_id = group[0]
            group_data = group[1]
            group_label = labels.iloc[i]

            data_path = 'X' + str(group_id)
            X.create_dataset(str(group_id), data=group_data)
            Y.create_dataset(str(group_id), data=group_label)

        self.f.close()


