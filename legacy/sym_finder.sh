#!/bin/bash
#$1 is poscar file $2 is number of atoms in uc
lat1=$(awk '{if(NR==3) print $0}' "$1")
lat2=$(awk '{if(NR==4) print $0}' "$1")
lat3=$(awk '{if(NR==5) print $0}' "$1")
declare -a pos 
x=9

y=$(($x + $2 - 1))
while [ $x -le $y ]
do
	pos[$x]=$(awk '{if(NR=='$x') print $0}' "$1")
	x=$(($x+1))
done
python atom_maker.py "$2"
atom=$(awk '{if(NR==1) print $0}' "atom_no.txt")

printf '%s\n' "Sample input file to identify_space_group.  BSO_1uc
0 
1
$lat1
$lat2
$lat3 
1 0 0
0 1 0
0 0 1
"$2"
"$atom"
"${pos[*]}"" |findsym
rm atom_no.txt

echo "Enter Filename:"
read arg1

mv findsym.log $arg1

