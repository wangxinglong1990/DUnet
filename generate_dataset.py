import tfbio
from tfbio.data import Featurizer
import pybel
import os
import numpy as np
import argparse

max_dist = 35
resolution = 2
scale = 0.5

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Data Generator Parameters')
    parser.add_argument('--database', type=str)

    args = parser.parse_args()

    output_path = 'dataset/'
    database_path = args.database

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    database_size = len(os.listdir(database_path))

    x_train = np.zeros((database_size, 18, 36, 36, 36), dtype=np.float32)
    y_train = np.zeros((database_size, 1, 36, 36, 36), dtype=np.float32)

    i = 0

    for filename in os.listdir(database_path):
        print(i, filename)
        prot = next(pybel.readfile('pdb', database_path + filename + '\\protein.pdb'))
        cavity = next(pybel.readfile('mol2', database_path + filename + '\\cavity6.mol2'))
        prot_featurizer = Featurizer(save_molecule_codes=False)
        cavity_featurizer = Featurizer(save_molecule_codes=False)
        prot_coords, prot_features = prot_featurizer.get_features(prot)
        cavity_coords, _ = cavity_featurizer.get_features(cavity)
        cavity_features = np.ones((cavity_coords.shape[0], 1), dtype=np.float32)
        centroid = prot_coords.mean(axis=0)
        prot_coords -= centroid
        cavity_coords -= centroid
        x = tfbio.data.make_grid(prot_coords, prot_features,
                                 max_dist=max_dist,
                                 grid_resolution=resolution)
        y = tfbio.data.make_grid(cavity_coords, cavity_features,
                                 max_dist=max_dist,
                                 grid_resolution=resolution)
        x = np.transpose(x, [0, 4, 1, 2, 3])
        y = np.transpose(y, [0, 4, 1, 2, 3])
        x_train[i] += x[0]
        y_train[i] += y[0]
        i += 1

    np.save(output_path + 'training_set.npy', x_train)
    np.save(output_path + 'test_set.npy', y_train)
