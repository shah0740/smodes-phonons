import numpy as np
from numpy.linalg import inv
import pandas as pd
import sys
import shutil

with open ( "POSCAR_superlat" , 'r') as file:
	lines=file.readlines()

lat_lines=lines[2:5]
pos_lines=lines[8:]

lat=np.asmatrix([line.strip().split() for line in lat_lines], dtype=float)
pos=np.asmatrix([line.strip().split() for line in pos_lines], dtype=float)
cart=pos*lat

disp_eig=[]

with open("Eigenvectors_Force_con", "r") as file:
	for i, line in enumerate(file):
		if line.startswith("Force Constants Eigenvector " + sys.argv[1] + ":"):
			for j in range(i+1,i+int(sys.argv[2])+1):
				disp_eig.append([float(x) for x in file.readline().strip().split()])	

disp=np.asmatrix(disp_eig)
disp_cart=cart+float(sys.argv[3])*disp
disp_frac=disp_cart*inv(lat)

shutil.copy("POSCAR_superlat","POSCAR_frozen")

with open(f"POSCAR_frozen", "r+") as fp:
	lines = fp.readlines()
	fp.seek(0)
	fp.truncate()
	fp.writelines(lines[:-int(sys.argv[2])])
	for line in disp_frac:
		np.savetxt(fp, line, fmt='%.15f')

