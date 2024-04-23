import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
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

def predict_center(file_location):
    temp = open(file_location, 'r')
    points = temp.read().splitlines()
    temp.close()
    x = []
    y = []
    z = []
    for point in points[7:]:
        if point[0] == '@':
            break
        x1, y1, z1 = [float(n) for n in point.split()[2:5]]
        x.append(x1)
        y.append(y1)
        z.append(z1)
    avg_x = np.mean(x)
    avg_y = np.mean(y)
    avg_z = np.mean(z)
    return np.array([avg_x, avg_y, avg_z])

def outputDCC_coach(out_file, predict_folder):
    c = 0
    folder = '/data/XLW/DUnet_train/coach_predict/curated_coach420/sites567/'
    for i in os.listdir(folder):
        f = open(r"%s%s/ligand.pdb" % (folder, i))
        ligand = f.readlines()
        f.close()

        x, y, z = [], [], []
        for ligand_atom in ligand:
            if ligand_atom.split()[0] == 'HETATM':
                x1, y1, z1 = [float(ligand_atom.split()[5]), float(ligand_atom.split()[6]),
                              float(ligand_atom.split()[7])]
                x.append(x1)
                y.append(y1)
                z.append(z1)
        avg_x = np.mean(x)
        avg_y = np.mean(y)
        avg_z = np.mean(z)
        center1 = np.array([avg_x, avg_y, avg_z])
        if os.path.exists('%s%s/0_out.mol2' % (predict_folder, i)):
            predict_pocket = predict_center('%s%s/0_out.mol2' % (predict_folder, i))
            dcc = np.linalg.norm(predict_pocket - center1)
            if float(dcc) <= 4:
                c += 1

    folder1 = '/data/XLW/DUnet_train/coach_predict/curated_coach420/sites678/'
    for i in os.listdir(folder1):
        f = open(r"%s%s/ligand.pdb" % (folder1, i))
        ligand = f.readlines()
        f.close()
        x, y, z = [], [], []
        for ligand_atom in ligand:
            if ligand_atom.split()[0] == 'HETATM':
                x1, y1, z1 = [float(ligand_atom.split()[6]), float(ligand_atom.split()[7]),
                              float(ligand_atom.split()[8])]
                x.append(x1)
                y.append(y1)
                z.append(z1)
        avg_x = np.mean(x)
        avg_y = np.mean(y)
        avg_z = np.mean(z)
        center1 = np.array([avg_x, avg_y, avg_z])
        if os.path.exists('%s%s/0_out.mol2' % (predict_folder, i)):
            predict_pocket = predict_center('%s%s/0_out.mol2' % (predict_folder, i))
            dcc = np.linalg.norm(predict_pocket - center1)
            # print(i,dcc)
            if float(dcc) <= 4:
                c += 1
    f = open('%s'%out_file,'a+')
    f.write('%s\n'%(str(int(c)/296)))
    f.close()

def outputDCC_BU48(out_file, predict_folder):
    c = 0
    folder = '/data/XLW/DUnet_train/test/bench2/'
    for i in os.listdir(folder):
        f = open(r"%s%s/ligand.mol2" % (folder, i))
        points = f.read().splitlines()
        f.close()

        x, y, z = [], [], []
        for point in points[7:]:
            if point[0] == '@':
                break
            x1, y1, z1 = [float(n) for n in point.split()[2:5]]
            x.append(x1)
            y.append(y1)
            z.append(z1)
        avg_x = np.mean(x)
        avg_y = np.mean(y)
        avg_z = np.mean(z)
        center = np.array([avg_x, avg_y, avg_z])
        if os.path.exists('%s%s/0_out.mol2' % (predict_folder, i)):
            predict_pocket = predict_center('%s%s/0_out.mol2' % (predict_folder, i))
            dcc = np.linalg.norm(predict_pocket - center)
            if float(dcc) <= 4:
                c += 1

    f = open('%s'%out_file,'a+')
    f.write('%s\n'%(str(int(c)/62)))
    f.close()

def outputDCC_sc6k(out_file, predict_folder):
    c = 0
    folder = '/data/XLW/DUnet_train/sc6k/curated/'
    for i in os.listdir(folder):
        f = open(r"%s%s/ligand.mol2" % (folder, i))
        ligand = f.readlines()
        f.close()

        x, y, z = [], [], []
        for ligand_atom in ligand[11:]:
            if ligand_atom[0] == '@':
                break
            x1, y1, z1 = [float(ligand_atom.split()[2]), float(ligand_atom.split()[3]),
                          float(ligand_atom.split()[4])]
            x.append(x1)
            y.append(y1)
            z.append(z1)
        avg_x = np.mean(x)
        avg_y = np.mean(y)
        avg_z = np.mean(z)
        center1 = np.array([avg_x, avg_y, avg_z])
        if os.path.exists('%s%s/0_out.mol2' % (predict_folder, i)):
            predict_pocket = predict_center('%s%s/0_out.mol2' % (predict_folder, i))
            dcc = np.linalg.norm(predict_pocket - center1)
            f = open('%s'%out_file, 'a+')
            f.write('%s %s\n' %(i,str(dcc)))
            f.close()
            if float(dcc) <= 4:
                c += 1

    f = open('%s'%out_file,'a+')
    f.write('%s\n'%(str(int(c)/2490)))
    f.close()

if __name__ == "__main__":
    device = torch.device('cuda:1')
    if not os.path.exists('dcc_epoch_bu48'):
        os.mkdir('dcc_epoch_bu48')
    bu48_set = '/data/XLW/DUnet_train/test/bench2/'
    for i in os.listdir('saved_model/'):
        if not os.path.exists('dcc_epoch_bu48/%s' %i):
            os.mkdir('dcc_epoch_bu48/%s' %i)
        model = DUnet.DenseNet().to(device)
        model.eval()
        model_path = "saved_model/%s"%i
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])
        for pro in os.listdir(bu48_set):
            prot = next(pybel.readfile('mol2', '%s%s/protein.mol2' % (bu48_set, pro)))
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
                if not os.path.exists('dcc_epoch_bu48/%s/%s/'%(i,pro)):
                  os.mkdir('dcc_epoch_bu48/%s/%s/'%(i,pro))
                p_mol.write('mol2', 'dcc_epoch_bu48/%s/%s/'%(i,pro) + '%s_out.mol2'%num, overwrite = True)
                num += 1

    f = open('dcc_check_bu48.txt','w')
    f.close()
    for e in os.listdir('saved_model/'):
        outputDCC_BU48('dcc_check_bu48.txt', 'dcc_epoch_bu48/%s/'%e)