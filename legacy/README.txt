Recipe to calculate phonons from SMODES

Prerequisites: 
1. POSCAR - delete all excess lines, so that atomic positions are the end of the file - use fractional coordinates
2. All the scripts present in the SMODES folder
3. POTCAR, INCAR (IBRION=-1), KPOINTS, VASP submit script

Procedure:
1. Use sym_finder to generate findym output, you will be asked to name the file

bash sym_finder.sh POSCAR $number_of_atoms_in_UC

2. Use generator_smodes_input.sh to generate smodes input, and output, for the desired k-vectors
If doing phonon calculations, only generate one k-vector smodes input/output at a time. 
If multiple k-vectors are present, sometimes errors occur in the next step. In fact
sometimes errors occur anyway in the next step, in which case redo step 2.

bash generator_smodes_input.sh #name_of_sym_file POSCAR

3. Use smodes_disp_calc.sh to generate displacements for the desired irrep - ensure
the k-vector was used in step 2

bash smodes_disp_calc.sh #name_of_smodes_output POSCAR #name_printed_to_POSCAR_disp_files

4. Go to the Irrep file and submit 
First make sure the KPOINTS and POSCAR_superlat are in agreement.
You will be prompted if the Wyckoff positions in the findsym output have degrees of freedom.
In which case enter y then an Undistorted lattice will be calculated, so that forces of the
Undistorted lattice will be removed from the Mode calculations.

bash sub.sh

5. After all displacement calculations are complete, force analysis will be done to
obtain eigenvalues and eigenvectors corresponding to the irrep.
Again you will be prompted if the Wyckoff positions in the findsym output have degrees of freedom.

bash force_analysis.sh 

