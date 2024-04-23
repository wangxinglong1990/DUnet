import os
import numpy as np
from Bio import PDB

def sc6k_sites():
    f = open('native_bind_sites_sc6k.txt','w')
    f.close()
    
    folder = '/data/XLW/DUnet_train/sc6k/curated_new/'
    for i in os.listdir(folder):
        f = open('native_bind_sites_sc6k.txt', 'a+')
        f.write('%s\n'%i)
        f.close()

        f = open(r"%s%s/ligand.mol2" % (folder, i))
        ligand = f.readlines()
        f.close()
        lig_coord = []
        for ligand_atom in ligand[11:]:
            if ligand_atom[0] == '@':
                break
            lig_coord.append(np.array([float(ligand_atom.split()[2]), float(ligand_atom.split()[3]),
                      float(ligand_atom.split()[4])]))

        f = open(r"%s%s/protein.pdb" % (folder, i))
        protein = f.readlines()
        f.close()

        seq = []
        binding_sites = []
        
        for protein_atom in protein:
            if protein_atom.split()[0] == 'ATOM':
                if protein_atom.split()[4] == "A1000":
                    break
                seq.append(int(protein_atom.split()[5]))
                pro_coord=np.array([float(protein_atom.split()[6]), float(protein_atom.split()[7]), float(protein_atom.split()[8])])
                for lig in lig_coord:
                    if np.linalg.norm(pro_coord - lig) <= 4:
                        binding_sites.append(int(protein_atom.split()[5]))
        sites = sorted(list(set(binding_sites)))
        full_lenth = sorted(list(set(seq)))
        f = open('native_bind_sites_sc6k.txt', 'a+')
        f.write('%s\n'%full_lenth)
        f.write('%s\n'%sites)
        f.close()

def coach_sites():
    f = open('native_bind_sites.txt','w')
    f.close()
    
    folder = '/data/XLW/DUnet_train/coach_predict/curated_coach420/sites567/'
    for i in os.listdir(folder):
        lig_coord = []
        f = open('native_bind_sites.txt', 'a+')
        f.write('%s\n'%i)
        f.close()

        f = open(r"%s%s/ligand.pdb" % (folder, i))
        ligand = f.readlines()
        f.close()

        for ligand_atom in ligand:
            if ligand_atom.split()[0] == 'HETATM':
                lig_coord.append(np.array(
                    [float(ligand_atom.split()[5]), float(ligand_atom.split()[6]), float(ligand_atom.split()[7])]))

        f = open(r"%s%s/protein.pdb" % (folder, i))
        protein = f.readlines()
        f.close()

        seq = []
        binding_sites = []
        for protein_atom in protein:
            if protein_atom.split()[0] == 'ATOM':
                seq.append(int(protein_atom.split()[5]))
                pro_coord=np.array([float(protein_atom.split()[6]), float(protein_atom.split()[7]), float(protein_atom.split()[8])])
                for lig in lig_coord:
                    if np.linalg.norm(pro_coord - lig) <= 4:
                        binding_sites.append(int(protein_atom.split()[5]))
        sites = sorted(list(set(binding_sites)))
        full_lenth = sorted(list(set(seq))) 
        f = open('native_bind_sites.txt', 'a+')
        f.write('%s\n'%full_lenth)
        f.write('%s\n'%sites)
        f.close()
        
    folder1 = '/data/XLW/DUnet_train/coach_predict/curated_coach420/sites678/'
    for i in os.listdir(folder1):
        lig_coord = []
        f = open('native_bind_sites.txt', 'a+')
        f.write('%s\n'%i)
        f.close()

        f = open(r"%s%s/ligand.pdb" % (folder1, i))
        ligand = f.readlines()
        f.close()

        for ligand_atom in ligand:
            if ligand_atom.split()[0] == 'HETATM':
                lig_coord.append(np.array(
                    [float(ligand_atom.split()[6]), float(ligand_atom.split()[7]), float(ligand_atom.split()[8])]))

        f = open(r"%s%s/protein.pdb" % (folder1, i))
        protein = f.readlines()
        f.close()

        seq = []
        binding_sites = []
        for protein_atom in protein:
            if protein_atom.split()[0] == 'ATOM':
                seq.append(int(protein_atom.split()[5]))
                pro_coord=np.array([float(protein_atom.split()[6]), float(protein_atom.split()[7]), float(protein_atom.split()[8])])
                for lig in lig_coord:
                    if np.linalg.norm(pro_coord - lig) <= 4:
                        binding_sites.append(int(protein_atom.split()[5]))
        sites = sorted(list(set(binding_sites)))
        full_lenth = sorted(list(set(seq))) 
        f = open('native_bind_sites.txt', 'a+')
        f.write('%s\n'%full_lenth)
        f.write('%s\n'%sites)
        f.close()

def predict_sites(predict_folder, predict_bind_sites):
    f = open(predict_bind_sites, 'w')
    f.close()
    folder = '/data/XLW/DUnet_train/sc6k/curated_new/'

    for i in os.listdir(predict_folder):
        if os.path.exists('%s%s/0_out.mol2' % (predict_folder, i)):
            f = open(predict_bind_sites, 'a+')
            f.write('%s\n'%i)
            f.close()

            f = open(r"%s%s/0_out.mol2" % (predict_folder, i))
            ligand = f.readlines()
            f.close()

            lig_coord = []
            for ligand_atom in ligand[7:]:
                if ligand_atom[0] == '@':
                    break

                lig_coord.append(np.array(
                    [float(ligand_atom.split()[2]), float(ligand_atom.split()[3]), float(ligand_atom.split()[4])]))

            if os.path.exists("%s%s/protein.pdb" % (folder, i)):
                f = open(r"%s%s/protein.pdb" % (folder, i))
                protein = f.readlines()
                f.close()

                seq = []
                binding_sites = []
                for protein_atom in protein:
                    if protein_atom.split()[0] == 'ATOM':
                        if protein_atom.split()[4] == "A1000":
                            break
                        seq.append(int(protein_atom.split()[5]))
                        pro_coord=np.array([float(protein_atom.split()[6]), float(protein_atom.split()[7]), float(protein_atom.split()[8])])
                        for lig in lig_coord:
                            if np.linalg.norm(pro_coord - lig) <= 4:
                                binding_sites.append(int(protein_atom.split()[5]))
                sites = sorted(list(set(binding_sites)))
                full_lenth = sorted(list(set(seq)))
                f = open(predict_bind_sites, 'a+')
                f.write('%s\n'%full_lenth)
                f.write('%s\n'%sites)
                f.close()