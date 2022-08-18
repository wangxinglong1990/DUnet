'''
Takes a PDB file and removes hetero atoms from its structure.
First argument is path to original file, second argument is path to generated file
'''
from Bio.PDB import PDBParser, PDBIO, Select
import Bio
import os
import sys
import pybel

database_dir = 'scPDB_classified/'


class NonHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if Bio.PDB.Polypeptide.is_aa(residue, standard=True) else 0


def clean_pdb(input_file, output_file):
    pdb = PDBParser().get_structure('protein', input_file)
    os.remove(input_file)
    io = PDBIO()
    io.set_structure(pdb)
    io.save(output_file, NonHetSelect())


if __name__ == '__main__':
    if __name__ == '__main__':
        count = 0
        for classification in os.listdir(database_dir):
            class_dir = database_dir + classification + '/'
            for name in os.listdir(class_dir):
                count += 1
                print(count, class_dir + name)
                protein = next(pybel.readfile('mol2', class_dir + name + '/' + 'protein.mol2'))
                protein.write('pdb', class_dir + name + '/' + 'protein.pdb')
                clean_pdb(class_dir + name + '/' + 'protein.pdb', class_dir + name + '/' + 'protein.pdb')
                os.remove(class_dir + name + '/' + 'protein.mol2')

