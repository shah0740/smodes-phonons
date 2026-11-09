#!/bin/bash

for file in Mode_*/
do
        cd $file/
#        cp ../KPOINTS ./
	sbatch submit_ankara_selfvasp_5.4.4.sh
        cd ..
	echo $file
done

echo "Are there degrees of freedom for the Wyckoff positions? (Enter y/n)"
read DOF_check
if [[ $DOF_check == "y" ]]; then
	mkdir Undistorted_lattice
	cp ../INCAR_fcon ../KPOINTS ../POTCAR ../submit_ankara_selfvasp_5.4.4.sh Undistorted_lattice
	cp POSCAR_superlat Undistorted_lattice/POSCAR
	cd Undistorted_lattice
	mv INCAR_fcon INCAR
#	cp ../KPOINTS ./
	sbatch submit_ankara_selfvasp_5.4.4.sh
	cd ..
else
	echo "No Undistorted_lattice calculated"
fi
