import os
import sys
import shutil

import torch
import DUnet
import pybel
from tfbio.data import Featurizer
import tfbio
import numpy as np
from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label
import openbabel
import argparse

from clean_pdb import clean_pdb

max_dist = 35
resolution = 2
scale = 0.5

binding_site_dist = 4


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

    parser = argparse.ArgumentParser(description='Prediction Parameters')
    parser.add_argument('--protein', type=str)
    parser.add_argument('--model', type=str, default='DUnet-3.pth')

    args = parser.parse_args()
    protein_file = args.protein
    model_path = args.model

    device = torch.device('cuda')
    model = DUnet.DenseNet().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])

    output_dir = 'result/' + protein_file[:-3] + '/'
    if not os.path.exists('result/'):
        os.mkdir('result/')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    clean_pdb(protein_file, protein_file)
    prot = next(pybel.readfile('pdb', protein_file))
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
        p_mol.write('mol2', output_dir + str(num) + '_out.mol2')
        print('pocket '+str(num)+' calculated')
        protein = prot
        pocket = p_mol

        binding_site = protein.clone

        amino_acid = set()

        prot_atoms = protein.atoms
        pock_atoms = pocket.atoms

        for i in range(0, len(protein.atoms)):
            for j in range(0, len(pocket.atoms)):
                protein_coord = np.array([float(n) for n in prot_atoms[i].coords])
                pocket_coord = np.array([float(n) for n in pock_atoms[j].coords])

                distance = np.linalg.norm(protein_coord - pocket_coord)

                if distance < 4:
                    amino_acid.add(protein.atoms[i].residue.idx)

        for i in range(len(protein.atoms), 0, -1):
            if not protein.atoms[i - 1].residue.idx in amino_acid:
                atom = binding_site.OBMol.GetAtom(i)
                binding_site.OBMol.DeleteAtom(atom)

        binding_site.write('pdb', output_dir + str(num) + '_binding_site.pdb')
        print('binding site '+str(num)+' calculated')
        num += 1
