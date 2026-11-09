import sys
with open("atom_no.txt", "a") as fp:
    for i in range(0,int(sys.argv[1])):
        fp.write("1 ")
