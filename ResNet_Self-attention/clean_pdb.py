from Bio.PDB import PDBParser, PDBIO, Select
import Bio
import os
import sys
#import pybel


class NonHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if Bio.PDB.Polypeptide.is_aa(residue, standard=True) else 0


def clean_pdb(input_file, output_file):
    pdb = PDBParser().get_structure('protein', input_file)
    os.remove(input_file)
    io = PDBIO()
    io.set_structure(pdb)
    io.save(output_file, NonHetSelect())

