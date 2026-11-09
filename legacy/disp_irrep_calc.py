import numpy as np
from numpy.linalg import inv
import pandas as pd
import sys
import csv
import shutil
from collections import Counter

f = open ( "Superlattice_vectors" , 'r')
l = []
l = [ line.split() for line in f]
f.close()
f = open ( "Superlattice_vectors" , 'r')
lines = []
lines= [line.split()[2:5] for line in f.readlines()[5:]]
f.close()

superlat=np.asmatrix(l[1:4],dtype=np.float64)
superlat_pos=np.asmatrix(lines,dtype=np.float64)
superlat_frac=superlat_pos*np.linalg.inv(superlat)

shutil.copy("Superlattice_vectors","Superlattice_frac")

with open("Superlattice_frac", 'r') as file:
    lines = file.readlines()

with open("Superlattice_frac", 'w') as file:
    file.writelines(lines[0:5])
    for line in superlat_frac:
        np.savetxt(file, line, fmt='%.7f')


unique_values_lattice=[]
with open ( "POSCAR" , 'r') as file:
	lines_lat=file.readlines()
	for line in lines_lat[2:5]:
		values = line.split()
		for value in values:
			float_value=float(value)
			if float_value not in unique_values_lattice:
				unique_values_lattice.append(float_value)

unique_values_pos=[]
with open ( "POSCAR" , 'r') as file:
        lines_lat=file.readlines()
        for line in lines_lat[8:]:
                values = line.split()
                for value in values:
                        float_value=float(value)
                        if float_value not in unique_values_pos:
                                unique_values_pos.append(float_value)

potential_dividers=np.array([2,3,4,0.5,1/3,0.25,5,0.2])
potential_multipliers=np.array([0.5,1/3,0.25,2,3,4,0.2,5])
threshold=0.0001
threshold_div=0.01    
for i in range(len(unique_values_lattice)):
	for j in range(superlat.shape[0]):
		for k in range(superlat.shape[1]):
			if unique_values_lattice[i] == 0:
				pass	
			elif abs(abs(unique_values_lattice[i])-abs(superlat[j,k])) < threshold:
				temp=superlat[j,k]
				if unique_values_lattice[i]==0:
					superlat[j,k]=unique_values_lattice[i]
				else:
                                        superlat[j,k]=unique_values_lattice[i]*round(temp/unique_values_lattice[i])
			else:
				for l in range(potential_dividers.shape[0]):
					if unique_values_lattice[i] !=0 and abs(abs(superlat[j,k]/unique_values_lattice[i])-potential_dividers[l]) < threshold_div \
					and len(str(superlat[j,k]).split('.')[1]) < 7:
						temp=superlat[j,k]*potential_dividers[l]
						if abs(round(temp/unique_values_lattice[i])) > 0:
							superlat[j,k]=unique_values_lattice[i]*round(temp/unique_values_lattice[i])*potential_multipliers[l]
						else:
							temp=superlat[j,k]
							superlat[j,k]=unique_values_lattice[i]*round(unique_values_lattice[i]/temp)*potential_dividers[l]*potential_dividers[l]
                                                    
common_values_pos=[0.1250000000000000,0.2500000000000000,0.3750000000000000,0.5000000000000000,0.6250000000000000,0.7500000000000000,0.87500000000000000,\
                 1.0000000000000,0.333333333333333,0.6666666666666667,0.1666666666666666,0.833333333333333]
threshold=0.0001

for i in range(len(unique_values_pos)):
        for j in range(superlat_frac.shape[0]):
                for k in range(superlat_frac.shape[1]):
                        if unique_values_pos[i] == 0:
                            continue
                        elif (abs(superlat_frac[j,k])-0.0000000000000000) < threshold:
                                superlat_frac[j,k]=0.0000000000000000
                        elif (abs(superlat_frac[j,k])-0.0000000000000000) > threshold:
                            if unique_values_pos[i] ==0 and unique_values_pos[i] ==0.5 and unique_values_pos[i] ==1:
                                continue
                            elif abs(unique_values_pos[i]-superlat_frac[j,k])<threshold or abs(unique_values_pos[i]+superlat_frac[j,k])<threshold:
                                temp=superlat_frac[j,k]
                                superlat_frac[j,k]=unique_values_pos[i]*round(temp/unique_values_pos[i])
                            elif abs((unique_values_pos[i]/2)-superlat_frac[j,k])<threshold or abs((unique_values_pos[i]/2)+superlat_frac[j,k])<threshold:
                                temp=superlat_frac[j,k]
                                superlat_frac[j,k]=(unique_values_pos[i]/2)*round(temp/(unique_values_pos[i]/2)) 
                            elif abs((unique_values_pos[i]/3)-superlat_frac[j,k])<threshold or abs((unique_values_pos[i]/3)+superlat_frac[j,k])<threshold:
                                temp=superlat_frac[j,k]
                                superlat_frac[j,k]=(unique_values_pos[i]/3)*round(temp/(unique_values_pos[i]/3)) 
                            elif abs((unique_values_pos[i]/4)-superlat_frac[j,k])<threshold or abs((unique_values_pos[i]/4)+superlat_frac[j,k])<threshold:
                                temp=superlat_frac[j,k]
                                superlat_frac[j,k]=(unique_values_pos[i]/4)*round(temp/(unique_values_pos[i]/4)) 
                            elif abs((unique_values_pos[i]/5)-superlat_frac[j,k])<threshold or abs((unique_values_pos[i]/5)+superlat_frac[j,k])<threshold:
                                temp=superlat_frac[j,k]
                                superlat_frac[j,k]=(unique_values_pos[i]/5)*round(temp/(unique_values_pos[i]/5))                                 
                            elif abs((1-unique_values_pos[i])-superlat_frac[j,k]) < threshold or abs((1-unique_values_pos[i])+superlat_frac[j,k]) < threshold:
                                temp=superlat_frac[j,k]
                                superlat_frac[j,k]=(1-unique_values_pos[i])*round(temp/(1-unique_values_pos[i]))
                            elif abs((1-unique_values_pos[i]/2)-superlat_frac[j,k]) < threshold or abs((1-unique_values_pos[i]/2)+superlat_frac[j,k]) < threshold:
                                temp=superlat_frac[j,k]
                                superlat_frac[j,k]=(1-unique_values_pos[i]/2)*round(temp/(1-unique_values_pos[i]/2))
                            elif abs(((1-unique_values_pos[i])/2)-superlat_frac[j,k]) < threshold or abs(((1-unique_values_pos[i])/2)+superlat_frac[j,k]) < threshold:
                                temp=superlat_frac[j,k]
                                superlat_frac[j,k]=((1-unique_values_pos[i])/2)*round(temp/((1-unique_values_pos[i])/2)) 
                            elif abs((unique_values_pos[i]*2)-superlat_frac[j,k])<threshold or abs((unique_values_pos[i]*2)+superlat_frac[j,k])<threshold:
                                temp=superlat_frac[j,k]
                                superlat_frac[j,k]=(unique_values_pos[i]*2)*round(temp/(unique_values_pos[i]*2)) 
                            elif abs(abs(1-(unique_values_pos[i]*2))-superlat_frac[j,k])<threshold or abs(abs(1-(unique_values_pos[i]*2))+superlat_frac[j,k])<threshold:
                                temp=superlat_frac[j,k]
                                superlat_frac[j,k]=(1-(unique_values_pos[i]*2))*round(temp/(1-(unique_values_pos[i]*2)))                                 
                            else:
                                for l in range(len(common_values_pos)):
                                    if abs(abs(superlat_frac[j,k])-common_values_pos[l]) < threshold:
                                        temp=superlat_frac[j,k]
                                        superlat_frac[j,k]=common_values_pos[l]*round(temp/common_values_pos[l])                                    
                                                            

with open("Superlattice_vectors") as file:
    # Skip the first 5 lines
    for i in range(5):
        next(file)

    # Count the occurrences of different strings in the 2nd column
    counter = Counter(line.split()[1] for line in file)
element_names=list(counter.keys())
element_nos=list(counter.values())


shutil.copy("POSCAR","POSCAR_superlat")

with open("POSCAR_superlat", "w") as fp:
	pass

with open("POSCAR_superlat" ,"w") as file:
	file.write(sys.argv[1] + "\n")
	file.write("1.0\n")
	for line in superlat:
        	np.savetxt(file, line, fmt='%.15f')

	file.write(' '.join(element_names) + "\n")
	file.write(' '.join(str(v) for v in element_nos) + "\n")
	file.write("Direct\n")

	for line in superlat_frac:
		np.savetxt(file, line, fmt='%.15f')

positions=superlat_frac*superlat
superlat_size=superlat_frac.shape[0]

no_of_modes=int(sys.argv[2])
for i in range(no_of_modes):
	shutil.copy("POSCAR_superlat",f"POSCAR_{i}")
	f = open ( f"Irrep_disp_{i}" , 'r')
	l = []
	l = [ line.split() for line in f]
	f.close()

	irrep_disp=np.asmatrix(l,dtype=np.float64)
	irrep_normed=irrep_disp/np.linalg.norm(irrep_disp)
	positions_disp=positions+0.001*irrep_normed
	irrep_disp_frac=positions_disp*np.linalg.inv(superlat)

	with open(f"POSCAR_{i}", "r+") as fp:
	    lines = fp.readlines()
	    fp.seek(0)
	    fp.truncate()
	    fp.writelines(lines[:-int(superlat_size)])

	    for line in irrep_disp_frac:
        	np.savetxt(fp, line, fmt='%.15f')
	
