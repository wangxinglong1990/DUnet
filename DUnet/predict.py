import torch
import numpy as np
from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label
import os

import DUnet
from clean_pdb import clean_pdb
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


def output_predict(binding_file, protein, predict_bind_sites):
    f = open(predict_bind_sites, 'w')
    f.close()

    if os.path.exists(binding_file):
        f = open(r'%s' % binding_file)
        ligand = f.readlines()
        f.close()

        lig_coord = []
        for ligand_atom in ligand[7:]:
            if ligand_atom[0] == '@':
                break
            lig_coord.append(np.array(
                [float(ligand_atom.split()[2]), float(ligand_atom.split()[3]), float(ligand_atom.split()[4])]))

        f = open(r"%s" % protein)
        protein = f.readlines()
        f.close()

        seq = []
        binding_sites = []
        for protein_atom in protein:
            if protein_atom.split()[0] == 'ATOM':
                seq.append(int(protein_atom.split()[5]))
                pro_coord = np.array([float(protein_atom.split()[6]), float(protein_atom.split()[7]),
                                      float(protein_atom.split()[8])])
                for lig in lig_coord:
                    dcc = np.linalg.norm(pro_coord - lig)

                    if dcc <= 4:
                        binding_sites.append(int(protein_atom.split()[5]))
        sites = sorted(list(set(binding_sites)))
        full_lenth = sorted(list(set(seq)))
        f = open(predict_bind_sites, 'a+')
        f.write('protein_sequence:\n%s\nbinding_sites\n' % full_lenth)
        for i in sites:
            f.write('%s\n' %i)
        f.close()


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='1.pdb')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-ofolder', type=str, default='output')

    args = parser.parse_args()
    infile = args.i
    out_folder = args.ofolder
    device = torch.device(args.device)

    outpath = out_folder
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    model = DUnet.DenseNet().to(device)
    model.eval()
    model_path = "epoch_14.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])

    prot = next(pybel.readfile('pdb', '%s' % infile))
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

        p_mol.write('mol2', '%s/%s_out.mol2' % (outpath, num), overwrite=True)
        output_predict('%s/%s_out.mol2' % (outpath, num), infile, '%s.txt' % infile)
        num += 1

