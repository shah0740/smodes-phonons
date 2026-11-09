#!/bin/bash

#$1 is the smodes output file
#$2 is the POSCAR file
#$3 is the title of file
#$mode is the irrep to be treated - make sure when making the smodes output file that the right k vector is being treated

echo "Enter irrep to be treated (ensure SMODES output contains the k vector):"
read mode
irrep_line=$(grep -n "$mode" $1 | cut -d: -f1)
degeneracy_line=$((irrep_line+1))
total_line=$((irrep_line+2))
degeneracy_irrep=$(sed -n "${degeneracy_line}p" $1 | awk '{print $2}')
total_modes=$(sed -n "${total_line}p" $1 | awk '{print $5}')

no_of_modes=$(($total_modes/$degeneracy_irrep))

lat_mat_line_pos=$(sed -n "${irrep_line},\$p" $1 | grep -n -m 1 "Vectors defining superlattice" | cut -d ':' -f 1)
pos_mat_line_pos=$(sed -n "${irrep_line},\$p" $1 | grep -n -m 1 "atom  type   position" | cut -d ':' -f 1)
sym_mode_line_pos=$(sed -n "${irrep_line},\$p" $1 | grep -n -m 1 "Symmetry modes:" | cut -d ':' -f 1)

lat_mat_line=$((irrep_line+lat_mat_line_pos-1))
pos_mat_line=$((irrep_line+pos_mat_line_pos-1))
sym_mode_line=$((irrep_line+sym_mode_line_pos-1))

echo "Vectors defining superlattice" >> Superlattice_vectors
sed -n "$((lat_mat_line+1)),$((lat_mat_line+3))p" $1 >> Superlattice_vectors
echo "Atom  Type   Position" >> Superlattice_vectors
sed -n "$((pos_mat_line+1)),$((sym_mode_line-1))p" $1 >> Superlattice_vectors

disp_arrow_line=$(sed -n "${irrep_line},\$p" $1 | grep -n -m ${no_of_modes} -- "---------------------" | cut -d ':' -f 1)
disp_start_line=$(echo $disp_arrow_line | awk '{for (i=1;i<=NF;i++) $i+=1}1')

k_vec_line=$(sed -n "${total_line},\$p" $1 | grep -n -m 1 "k vector" | cut -d ':' -f 1)
irrep_line_check=$(sed -n "${total_line},\$p" $1 | grep -n -m 1 "Irrep" | cut -d ':' -f 1)

if [ -n "$k_vec_line" ] && [ $k_vec_line -lt $irrep_line_check ]; then
	disp_arrow_end_line=$(sed -n "${total_line},\$p" $1 | grep -n -m 1 "k vector" | cut -d ':' -f 1)
	disp_final_line=$((disp_arrow_end_line))
	
elif (sed -n "${total_line},\$p" $1 | grep -q -m 1 "Irrep"); then
        disp_arrow_end_line=$(sed -n "${total_line},\$p" $1 | grep -n -m 1 "Irrep" | cut -d ':' -f 1)
        disp_final_line=$((disp_arrow_end_line))
	
else
       	disp_arrow_end_line=$(sed -n "${total_line},\$p" $1 | grep -n -m 1 "End of output" | cut -d ':' -f 1)
        disp_final_line=$((disp_arrow_end_line))
	
fi

disp_endings=$disp_start_line

read -ra arr <<< "$disp_endings"
len=${#arr[@]}

if (( len == 1)); then 
	disp_end_line=$disp_final_line
else 
	arr=("${arr[@]:1:$len}")
	for i in "${!arr[@]}" ; do
		arr[i]=$((${arr[i]}-2))
	done
	disp_ends="${arr[@]}"

	disp_end_line="${disp_ends} ${disp_final_line}"
fi

IFS=" " read -r -a disp_start <<< "$disp_start_line"
IFS=" " read -r -a disp_end <<< "$disp_end_line"

for ((i=0; i<no_of_modes; i++)); do
        sed -n "${irrep_line},\$p" $1 | sed -n "${disp_start[$i]},${disp_end[$i]}p" > Irrep_mode_$i
done

superlat_size=$(tail -n 1 Superlattice_vectors | awk '{print $1}')

for i in $(seq 0 $((no_of_modes-1))); do
	for k in $(seq 1 $superlat_size); do
		awk -v num="$k" '$1==num {print $(NF-2), $(NF-1), $NF; found=1; exit} END{if(!found) print "0.00000 0.00000 0.00000"}' Irrep_mode_$i
	done > Irrep_disp_$i

        ele_name_irrep=$(awk 'NR==1 {print $2}' Irrep_mode_$i)

        echo "What is the atomic mass of $ele_name_irrep:"
        read atomic_mass_ele

        echo "Atomic mass of $ele_name_irrep: $atomic_mass_ele" >> Irrep_mode_$i
	
done
mkdir Irrep_$mode
mv Superlattice_vectors Irrep_$mode
mv Irrep_mode_* Irrep_$mode
mv Irrep_disp_* Irrep_$mode
cp POSCAR KPOINTS INCAR_relax disp_irrep_calc.py sub.sh force_analysis.sh phonon_modes.py fcon_phonon.py freeze_phonons.sh Irrep_$mode

cd Irrep_$mode

for i in $(seq 0 $((no_of_modes-1))); do
	mkdir Mode_$i
	cp ../KPOINTS ../POTCAR ../INCAR_fcon ../submit_ankara_selfvasp_5.4.4.sh Mode_$i
	mv Mode_$i/INCAR_fcon Mode_$i/INCAR
done

python disp_irrep_calc.py $3 $no_of_modes

for i in $(seq 0 $((no_of_modes-1))); do
	mv Irrep_disp_$i Irrep_mode_$i POSCAR_$i Mode_$i
	cd Mode_$i
	mv POSCAR_$i POSCAR
	cd ..
done
