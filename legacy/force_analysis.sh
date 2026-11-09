#!/bin/bash

superlat_size=$(wc -l < Mode_0/Irrep_disp_0)
mkdir Force_data
cp phonon_modes.py Force_data
cp POSCAR Force_data/Phonon_frequencies
cp POSCAR Force_data/Eigenvalues_fcon
cp POSCAR Force_data/Eigenvectors_Dyn_mat
cp POSCAR Force_data/Eigenvectors_Force_con
cp POSCAR Force_data/Eigenvectors_Atom_disp


i=0
for mode in Mode_*
do
	cd $mode
	gzip -d OUTCAR.gz
	total_force_line=$(grep -n "TOTAL-FORCE" OUTCAR | cut -d: -f1)
	force_line_start=$((total_force_line+2))

	sed -n "$((force_line_start)),$((force_line_start+superlat_size-1))p" OUTCAR | awk '{print $(NF-2), $(NF-1), $NF}' > force_raw_output_$mode
	cp force_raw_output_$mode Irrep_* ../Force_data
	i=$((i+1))
	cd ..
done

echo "Are there degrees of freedom for the Wyckoff positions? (Enter y/n)"
read DOF_check
if [[ $DOF_check == "y" ]]; then
	cd Undistorted_lattice
        gzip -d OUTCAR.gz
	total_force_line=$(grep -n "TOTAL-FORCE" OUTCAR | cut -d: -f1)
        force_line_start=$((total_force_line+2))
        sed -n "$((force_line_start)),$((force_line_start+superlat_size-1))p" OUTCAR | awk '{print $(NF-2), $(NF-1), $NF}' > force_raw_output_undistorted
        cp force_raw_output_undistorted ../Force_data
	cd ..

	cd Force_data
	python phonon_modes.py $i 1
	cd ..

else
        cd Force_data
	python phonon_modes.py $i
	cd ..
fi

cp Force_data/Eigenvectors_* Force_data/Phonon_frequencies Force_data/Eigenvalues_fcon ./

