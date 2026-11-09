import numpy as np
from numpy.linalg import inv
import pandas as pd
from numpy.linalg import eig
import sys
import csv
import shutil

e=1.602177E-19
perm=8.854187E-12
c = 3E10
AMU = 1.66053E-27
CONST=e/(1E-10)**2

with open("force_raw_output_Mode_0", 'r') as fp:
	row_size=len(fp.readlines())

for i in range(int(sys.argv[1])):
	f = open ( f"force_raw_output_Mode_{i}" , 'r')
	l = []
	l = [ line.split() for line in f]
	f.close()
	vars()[f"forces{i}"]=np.asmatrix(l,dtype=np.float64)
	
for i in range(int(sys.argv[1])):
	f = open ( f"Irrep_disp_{i}" , 'r')
	l = []
	l = [ line.split() for line in f]
	f.close()
	vars()[f"basis{i}"]=np.asmatrix(l,dtype=np.float64)  

mass_list=[]
mass_arr=[[0 for col in range(int(sys.argv[1]))] for row in range(int(sys.argv[1]))]
mass_mat=[[0 for col in range(int(sys.argv[1]))] for row in range(int(sys.argv[1]))]

for i in range(int(sys.argv[1])):
	with open(f"Irrep_mode_{i}" , 'r') as f:
		last_line=f.readlines()[-1]
		last_value = float(last_line.split(':')[-1].strip()) 
		mass_list.append(last_value)

for i in range(int(sys.argv[1])):
	for j in range(int(sys.argv[1])):
		mass_arr[i][j]=np.sqrt(mass_list[j])

mass_mat=np.multiply(mass_arr,np.transpose(mass_arr))

FFF=[[0 for col in range(int(sys.argv[1]))] for row in range(int(sys.argv[1]))]

if len(sys.argv) == 2:
	for i in range(int(sys.argv[1])):
		for j in range(int(sys.argv[1])):
			force_multi=np.multiply((vars()[f"basis{j}"]/np.linalg.norm(vars()[f"basis{j}"])),vars()[f"forces{i}"])
			FFF[i][j]=force_multi.sum()
else: 
	f = open ( f"force_raw_output_undistorted" , 'r')
	l = []
	l = [ line.split() for line in f]
	f.close()
	forces_undistorted=np.asmatrix(l,dtype=np.float64)
	for i in range(int(sys.argv[1])):
		for j in range(int(sys.argv[1])):
			force_multi=np.multiply((vars()[f"basis{j}"]/np.linalg.norm(vars()[f"basis{j}"])),(vars()[f"forces{i}"]-forces_undistorted))
			FFF[i][j]=force_multi.sum()

disp_coeff=0.001
D=-np.divide(FFF,disp_coeff)

force_mat=np.divide((D+np.transpose(D)),2)
dyn_mat=np.divide(force_mat,mass_mat)

force_eig,force_vec=eig(force_mat)
dyn_eig,dyn_vec=eig(dyn_mat)

wTO=np.power(np.divide(np.multiply(dyn_eig,CONST),AMU),0.5,dtype=complex)
muTO=np.diag(wTO/(2*np.pi*c))

eigenvalues=np.diag(muTO)
eig_cm=eigenvalues
eig_mev=eig_cm/8.066
eig_thz=eig_cm/33.356
       
eigenvalues_stack=np.column_stack((eig_cm,eig_thz,eig_mev))

with open("Phonon_frequencies", "w") as fp:
        pass

with open("Phonon_frequencies" ,"w") as file:
	file.write("Phonon Frequencies\n")
	file.write('cm-1'.rjust(7) + 'THz'.rjust(15) + 'meV'.rjust(15) + '\n')
	for i in range(len(eig_cm)):
		file.write("{:.2f}+{:.2f}j\t{:.2f}+{:.2f}j\t{:.2f}+{:.2f}j\n".format(eig_cm[i].real,eig_cm[i].imag,eig_thz[i].real,eig_thz[i].imag,eig_mev[i].real,eig_mev[i].imag))	

force_eig_ev_a2=np.sqrt(force_eig,dtype=complex)

with open("Eigenvalues_fcon", "w") as fp:
        pass

with open("Eigenvalues_fcon" ,"w") as file:
        file.write("Force Constant Matrix Eigenvalues\n")
        file.write('eV/A^2'.rjust(7) + '\n')
        for i in range(len(force_eig_ev_a2)):
                file.write("{:.2f}+{:.2f}j\n".format(force_eig_ev_a2[i].real,force_eig_ev_a2[i].imag))

with open("Eigenvectors_Force_con", "w") as fp:
        pass
with open("Eigenvectors_Dyn_mat", "w") as fp:
        pass
with open("Eigenvectors_Atom_disp", "w") as fp:
        pass

size=(row_size,3)

for i in range(int(sys.argv[1])):
	vars()[f"force_con_eig{i}"]=0*np.ones(size)
	for j in range(int(sys.argv[1])):
		vars()[f"force_con_eig{i}"]=vars()[f"force_con_eig{i}"]+force_vec[j][i]*vars()[f"basis{j}"]
	vars()[f"force_con_eig{i}"]=vars()[f"force_con_eig{i}"]/np.linalg.norm(vars()[f"force_con_eig{i}"])
	
with open("Eigenvectors_Force_con", "w") as fp:
	fp.write("Force Constant Matrix")
	for i in range(int(sys.argv[1])):
		fp.write(f"\nForce Constants Eigenvector {i}:\n")
		np.savetxt(fp,vars()[f"force_con_eig{i}"], fmt="%.7f")

	
dyn_eig_real=np.divide(dyn_vec,np.transpose(mass_arr))

for i in range(int(sys.argv[1])):
	vars()[f"dyn_eig{i}"]=0*np.ones(size)
	for j in range(int(sys.argv[1])):
                vars()[f"dyn_eig{i}"]= vars()[f"dyn_eig{i}"]+dyn_vec[j][i]*vars()[f"basis{j}"]
	vars()[f"dyn_eig{i}"]=vars()[f"dyn_eig{i}"]/np.linalg.norm(vars()[f"dyn_eig{i}"])

with open("Eigenvectors_Dyn_mat", "w") as fp:
        fp.write("Dynamical Matrix")
        for i in range(int(sys.argv[1])):
                fp.write(f"\nDynamical Matrix Eigenvector {i}:\n")
                np.savetxt(fp,vars()[f"dyn_eig{i}"], fmt="%.7f")


for i in range(int(sys.argv[1])):
	vars()[f"phonon_disp{i}"]=0*np.ones(size)
	for j in range(int(sys.argv[1])):
		vars()[f"phonon_disp{i}"]=vars()[f"phonon_disp{i}"]+dyn_eig_real[j][i]*vars()[f"basis{j}"]
	vars()[f"phonon_disp{i}"]=vars()[f"phonon_disp{i}"]/np.linalg.norm(vars()[f"phonon_disp{i}"])
with open("Eigenvectors_Atom_disp", "w") as fp:
        fp.write("Atomic Displacements")
        for i in range(int(sys.argv[1])):
                fp.write(f"\nPhonon Displacement {i}:\n")
                np.savetxt(fp,vars()[f"phonon_disp{i}"], fmt="%.7f")

