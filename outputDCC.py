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


def coach_center(file_location):
    temp = open(file_location, 'r')
    points = temp.read().splitlines()
    temp.close()
    x = []
    y = []
    z = []
    for point in points[2:]:
        if point.split()[0] == 'CONECT':
            break
        x1 = float(point[27:38])
        y1 = float(point[38:46])
        z1 = float(point[46:54])

        x.append(x1)
        y.append(y1)
        z.append(z1)
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        return np.array([10000, 10000, 10000])
    avg_x = np.mean(x)
    avg_y = np.mean(y)
    avg_z = np.mean(z)
    return np.array([avg_x, avg_y, avg_z])


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
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        return np.array([10000, 10000, 10000])
    avg_x = np.mean(x)
    avg_y = np.mean(y)
    avg_z = np.mean(z)
    return np.array([avg_x, avg_y, avg_z])


def outputDCC_coach(file, epoch, Path1, Path2):
    count = 0
    for pocket in os.listdir(Path1):
        try:
            actual_pocket = coach_center(Path1 +  pocket + '/ligand.pdb')
            predict_pocket = predict_center('%s/%s/0_out.mol2'%(Path2,pocket))
            dcc = np.linalg.norm(predict_pocket - actual_pocket)
            # f=open(file,'a+')
            # f.write(pocket+' '+str(dcc)+'\n')
            # f.close()
            if float(dcc) <= 4:
                count += 1
        except:
            pass

    f = open(file, 'a+')
    f.write('%s %s\n' % (epoch, count / 298))
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-device', type=str, default='cpu')
    
    args = parser.parse_args()
    epoch = args.epoch
    device = args.device

    device = torch.device(device)
    if not os.path.exists('dcc_epoch'):
        os.mkdir('dcc_epoch')
    if not os.path.exists('dcc_epoch/%s'%epoch):
        os.mkdir('dcc_epoch/%s'%epoch)
    coach_set = $validation_coach_set
    model = DUnet.DenseNet().to(device)
    model.eval()
    model_path = "model.pth" 
    #model_path = "saved_model/" + 'epoch_' + str(epoch) + '.pth'
    checkpoint = torch.load(model_path,map_location=lambda storage, loc:storage)
    model.load_state_dict(checkpoint['net'])
    for pro in os.listdir(coach_set):
        protein = coach_set + pro + '/protein.pdb'
        output_dir = 'dcc_epoch/%s/'%epoch + pro + '/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        prot = next(pybel.readfile('pdb', '%s%s/protein.pdb' % (coach_set, pro)))
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
            p_mol.write('mol2', output_dir + '%s_out.mol2' % num)
            num += 1
    outputDCC_coach('dcc_cpu.txt', epoch, coach_set, 'dcc_epoch/%s'%epoch)
