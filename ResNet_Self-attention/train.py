import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label
import os

import RS
from clean_pdb import clean_pdb
import pybel
import openbabel
import tfbio
from tfbio.data import Featurizer

dataset_path1 = $location_of_generated_dataset_1
dataset_path2 = $location_of_generated_dataset_2
coach_set = $Coach_validation_set_location

torch.manual_seed(42)

bz = 16

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


class DiceLoss(nn.Module):
    def init(self):
        super(DiceLoss, self).init()

    def forward(self, pred, target):
        smooth = 0.01
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)

        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


class MyDataset(Dataset):
    def __init__(self, x, y, start, end):
        self.x = x[start:end]
        self.y = y[start:end]
        self.data = 0
        self.label = 0

    def __getitem__(self, index):
        self.data = self.x[index]
        self.label = self.y[index]
        self.data = torch.from_numpy(self.data)
        return self.data, self.label

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = torch.device('cuda:1')
    f = open('coach_acc.txt', 'w')
    f.close()
    f = open('BU_acc.txt', 'w')
    f.close()
    if not os.path.exists('saved_model'):
        os.mkdir('saved_model')
    if not os.path.exists('coach_results'):
        os.mkdir('coach_results')
    if not os.path.exists('BU_results'):
        os.mkdir('BU_results')
        
    #clean_pdb('coach420/%s/protein.pdb' % pro, 'coach420/%s/protein.pdb' % pro)
    
    x = np.load(dataset_path1 + 'x.npy')
    y = np.load(dataset_path1 + 'y.npy')
    x1 = np.load(dataset_path2 + 'x.npy')
    y1 = np.load(dataset_path2 + 'y.npy')
    x_whole = np.concatenate((x,x1),0)
    y_whole = np.concatenate((y,y1),0)

    shuffle_index = np.arange(len(x))
    np.random.shuffle(shuffle_index)
    x_whole = x_whole[shuffle_index]
    y_whole = y_whole[shuffle_index]


    train_set = MyDataset(x_whole, y_whole, 0, len(x_whole))
    #train_set = MyDataset(x_whole, y_whole, 0, 20)
    x = None
    y = None

    model = RS.RS().to(device)
    train_loader = DataLoader(dataset=train_set, batch_size=bz, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    dice_loss = DiceLoss().to(device)

    losslist = []
    epochlist = []
    vallist = []
    for epoch in range(200):
        print('epoch %s' % epoch)
        model.train()
        epoch_loss = 0.0
        for i, data in enumerate(train_loader):
            x, y = data
            x, y = x.to(device), y.to(device)
            out = model(x)
            # out = torch.squeeze(out)
            loss = dice_loss(out, y)
            # dice_loss=criteria(out, y)
            # loss=dice_loss
            print(epoch, loss.item())
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch_loss / len(train_set))
        losslist.append(epoch_loss / len(train_set))
        epochlist.append(epoch)

        if epoch > 10 and (epoch+1)%3 == 0:
            state_dict = {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
            torch.save(state_dict, r"saved_model/" + 'epoch_' + str(epoch) + '.pth')
            
            model.eval()
            valid_loss = 0.0
            model_path = "saved_model/" + 'epoch_' + str(epoch) + '.pth'
            
            os.system('rm -rf coach_results/*')
            
            with torch.no_grad():
                checkpoint = torch.load(model_path)
                model.load_state_dict(checkpoint['net'])

                for pro in os.listdir(coach_set):
                    protein = coach_set + pro + '/protein.pdb'
                    output_dir = 'coach_results/' + pro + '/'
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
                outputDCC_coach('coach_acc.txt', epoch, coach_set, 'coach_results')


