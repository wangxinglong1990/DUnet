import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label
import os

import RS
import pybel
import openbabel
import tfbio
from tfbio.data import Featurizer
import argparse

max_dist = 35
resolution = 2
scale = 0.5


def get_pockets_segmentation(density, threshold=0.5, min_size=50):
    if len(density) != 1:
        raise ValueError('segmentation of more than one pocket is not'
                         ' supported')

    voxel_size = (1 / scale) ** 3
    bw = closing((density[0] > threshold).any(axis=-1))
    cleared = clear_border(bw)

    label_image, num_labels = label(cleared, return_num=True)
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum() * voxel_size
        if pocket_size < min_size:
            label_image[np.where(pocket_idx)] = 0

    return label_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-infile', type=str, default='1.pdb')
    parser.add_argument('-name', type=str, default='test')
    parser.add_argument('-device', type=str, default='cpu')
    parser.add_argument('-out_folder', type=str, default='predict_pocket')
    
    args = parser.parse_args()
    infile = args.infile
    name = args.name
    device = args.device
    out_folder = args.out_folder

    device = torch.device(device)
    output = '%s/%s'%(out_folder,name)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if not os.path.exists(output):
        os.mkdir(output)
    
    model = RS.RS().to(device)
    model.eval()
    model_path = "model.pth"
    checkpoint = torch.load(model_path,map_location=lambda storage, loc:storage)
    model.load_state_dict(checkpoint['net'])

    file = infile

    prot = next(pybel.readfile('pdb',file))
    prot_featurizer = Featurizer(save_molecule_codes=False)
    prot_coords, prot_features = prot_featurizer.get_features(prot)
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid
    origin = (centroid - max_dist)
    step = np.array([1.0 / scale] * 3)
    x = tfbio.data.make_grid(prot_coords, prot_features,
                             max_dist=max_dist,
                             grid_resolution=resolution)
    x = np.transpose(x, [0, 4, 1, 2, 3])
    protein = torch.from_numpy(x)
    protein = protein.to(device)
    out = model(protein)
    out = out.cpu().detach().numpy()
    out = np.transpose(out, [0, 2, 3, 4, 1])
    pockets = get_pockets_segmentation(out)

    num = 0
    for pocket_label in range(1, pockets.max() + 1):

        indices = np.argwhere(pockets == pocket_label).astype('float32')
        indices *= step
        indices += origin
        mol = openbabel.OBMol()
        for idx in indices:
            a = mol.NewAtom()
            a.SetVector(float(idx[0]), float(idx[1]), float(idx[2]))
        p_mol = pybel.Molecule(mol)
        p_mol.write('mol2', output + '/%s_out.mol2' % num)
        num += 1
