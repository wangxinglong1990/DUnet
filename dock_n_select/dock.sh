##first thing is to combine the ligand and protein structure file before docking
cat pro.pdb LIG_0001.pdb > complex.pdb
rosetta_scripts.mpi.linuxgccrelease -s complex.pdb -parser:protocol dock.xml -extra_res_fa LIG.params -nstruct 200 -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false -overwrite -ignore_unrecognized_res
