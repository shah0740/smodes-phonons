#!/bin/bash

echo "Which mode (number) of the irrep do you want to freeze in? Refer to the Eigenvalues_fcon file:"
read mode_no

total_lines=$(wc -l < POSCAR_superlat)
line=$((total_lines-8))

echo "What coefficient of the force constant eigenvectors do you want to freeze in?"
read coeff

python fcon_phonon.py $mode_no $line $coeff

mkdir Phonon_Frozen_mode_$mode_no_$coeff
mv POSCAR_frozen Phonon_Frozen_mode_$mode_no_$coeff/POSCAR
cp INCAR_relax Phonon_Frozen_mode_$mode_no_$coeff/INCAR
cp Mode_0/KPOINTS Mode_0/POTCAR Mode_0/submit_ankara_selfvasp_5.4.4.sh Phonon_Frozen_mode_$mode_no_$coeff
