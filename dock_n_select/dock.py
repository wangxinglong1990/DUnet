##Rosetta and gromacs should be installed
import os
import numpy as np
os.system('python predict.py -i 1.pdb')
f = open(r'output/0_out.mol')
ligand = f.readlines()
f.close()
lig_coord = []
for ligand_atom in ligand[7:]:
    if ligand_atom[0] == '@':
        break
    lig_coord.append(np.array(
        [float(ligand_atom.split()[2]), float(ligand_atom.split()[3]), float(ligand_atom.split()[4])]))
center = np.mean(lig_coord,axis=1)
os.system('gmx editconf -f lig.pdb -o ligc.pdb -center %s %s %s'%(float(center[0])/10,float(center[1])/10,float(center[2])/10))
os.system('obabel ligc.pdb -O LIG.mol2')
os.system('bash convert_qm2mol2.sh LIG.mol2 ')
os.system('ln -s $ROSETTA/main/source/scripts/python/public/molfile_to_params.py ./')
os.system('python molfile_to_params.py -n LIG --extra_torsion_output LIG.mol2')
os.system('cat 1.pdb LIG.pdb > complex.pdb')
os.system('rosetta_scripts.mpi.linuxgccrelease -s complex.pdb -parser:protocol dock.xml -extra_res_fa LIG.params -nstruct 200 -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false -overwrite -ignore_unrecognized_res')
