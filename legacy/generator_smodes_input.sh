#!/bin/bash
# the goal is to create a file called smodes_input which can be entered to obtain smodes information

# $1 is the findsymfile
# $2 is the POSCAR file

echo $1 >> smodes_input

line_params=$(grep "Lattice parameters, a,b,c,alpha,beta,gamma:" $1)
# Get the next line containing the lattice parameters
params=$(sed -n "/$line_params/{n;p;q}" $1)

#gets the space group #
space_group=$(grep "Space Group" $1 | awk '{print $3}')

#gets the number of wyckoff positions
num_wyckoff=$(grep -o "Wyckoff position" $1 | wc -l)

#prints first 4 lines of smodes input
echo $space_group >> smodes_input
echo $params >> smodes_input
echo $num_wyckoff >> smodes_input

#identifies lines containing the phrase wyckoff position
wyckoff_lines=$(grep -n "Wyckoff position" $1 | cut -f1 -d:)

#identifies the next line as that contains the atom number which will be used to identify atom name
for p in $wyckoff_lines; do
	wyckoff_atoms="$wyckoff_atoms $((p+1))"
done
IFS=" " read -r -a wyc_atom_phrases <<< "$wyckoff_atoms"

# for some reason (syntax?) we need to go reverse to obtain wyckoff position lines index
for p in $wyckoff_atoms; do
        wyckoff_li="$wyckoff_li $((p-1))"
done

IFS=" " read -r -a wyc_lines <<< "$wyckoff_li"

#opens POSCAR to find type and number of atoms
atom_list=$(sed -n '6p' $2)
atom_no=$(sed -n '7p' $2)
num_atom=$(echo $atom_list | awk -F' ' '{print NF}')

IFS=" " read -r -a atom_phrases <<< "$atom_list"
IFS=" " read -r -a atom_no_phrases <<< "$atom_no"

#prints atom name in a file called atom_index line by line, so that the atom number above can be replaced with the atom name 
for ((i=0; i<num_atom; i++)); do
	for ((k=0; k<"${atom_no_phrases[i]}" ; k++)); do
		echo "${atom_phrases[i]}" >> atom_index
	done
done

#here we loop over the number of types the term wyckoff position appears in the find sym file to write the smodes input
for ((i=0; i<num_wyckoff; i++)); do
	
	#referencing the line containing the wyckoff position info
	line_wyc=$(sed -n "${wyc_lines[i]}p" $1)
	
	# first check for if wyckoff position has independent variables
	if [ ${#line_wyc} -gt 18 ]; then
		#char is the wyckoff position
		char=$(echo "$line_wyc" | cut -c18)
		
		#first check if there are x y z independent variables
		if [[ $(echo $line_wyc | cut -c19- | grep -o "x") && $(echo $line_wyc | cut -c19- | grep -o "y") && $(echo $line_wyc | cut -c19- | grep -o "z") ]]; then
                        clean_line_wyc=$(echo "$line_wyc" | tr -d ',')
                        
                        x_val=$(echo "$clean_line_wyc" | awk '{print $6}')
                        y_val=$(echo "$clean_line_wyc" | awk '{print $9}')
                        z_val=$(echo "$clean_line_wyc" | awk '{print $12}')
		
		#next check if there are xy xz or yz combination of indepedent variables
		elif [[ $(echo $line_wyc | cut -c19- | grep -o "x") && $(echo $line_wyc | cut -c19- | grep -o "y") ]]; then
			clean_line_wyc=$(echo "$line_wyc" | tr -d ',')
			
			x_val=$(echo "$clean_line_wyc" | awk '{print $6}')
                        y_val=$(echo "$clean_line_wyc" | awk '{print $9}')
                        z_val=0
                elif [[ $(echo $line_wyc | cut -c19- | grep -o "x") && $(echo $line_wyc | cut -c19- | grep -o "z") ]]; then
                        clean_line_wyc=$(echo "$line_wyc" | tr -d ',')
                        
                        x_val=$(echo "$clean_line_wyc" | awk '{print $6}')
                        z_val=$(echo "$clean_line_wyc" | awk '{print $9}')
                        y_val=0
                elif [[ $(echo $line_wyc | cut -c19- | grep -o "y") && $(echo $line_wyc | cut -c19- | grep -o "z") ]]; then
                        clean_line_wyc=$(echo "$line_wyc" | tr -d ',')
 
                        y_val=$(echo "$clean_line_wyc" | awk '{print $6}')
                        z_val=$(echo "$clean_line_wyc" | awk '{print $9}')
                        x_val=0
                else 
			# if only one of x y z is present, and we still need to write 0 in the smodes input
                        if [[ ${line_wyc:18} == *"x"* ]]; then
                                x_val=$(echo "$line_wyc" | awk '{print $6}')
                                y_val=0
                                z_val=0
                        elif [[ ${line_wyc:18} == *"y"* ]]; then
                                x_val=0
                                y_val=$(echo "$line_wyc" | awk '{print $6}')
                                z_val=0
                        else
                                x_val=0
                                y_val=0
                                z_val=$(echo "$line_wyc" | awk '{print $6}')
                        fi

		fi
		
		#here we use the atom number in the line after wyckoff position and use the atom_index file to substitute atom name
                line_atom=$(sed -n "${wyc_atom_phrases[i]}p" $1)
                ele_index=$(echo "$line_atom" | awk '{print $1}')
                ele_name=$(sed -n "${ele_index}p" atom_index)
	
		#writing all the wyckoff information including atom name wyckoff pos and the independent variables 
                echo $ele_name $char $x_val $y_val $z_val >> smodes_input

	else 
		#this is if the wyckoff position has no independent variables
		char=$(echo "$line_wyc" | cut -c18)
	
		line_atom=$(sed -n "${wyc_atom_phrases[i]}p" $1)
		ele_index=$(echo "$line_atom" | awk '{print $1}')
		ele_name=$(sed -n "${ele_index}p" atom_index)
		echo $ele_name $char >> smodes_input
	
	fi

done

echo "Enter the number of k-vectors to be treated"
read count

echo $count >> smodes_input

for ((i=1;i<=count;i++)); do
	echo "Enter the name of k-vectors to be treated"
	read arg
	echo $arg >> smodes_input
done
#echo "Enter the name of the k-vectors to be treated"

# here atom_index is removed, but comment out below line to verify
rm atom_index

mv smodes_input $1_smodes_input

smodes < $1_smodes_input >> $1_smodes_output
