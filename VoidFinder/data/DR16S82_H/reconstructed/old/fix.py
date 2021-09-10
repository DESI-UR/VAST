import shutil

filename = 'data_reconstructed_removed.dat'

from_file = open(filename) 
line = from_file.readline()

# make any changes to line here

line="# ra dec z delta\n"

to_file = open(filename,mode="w")

to_file.write(line)

shutil.copyfileobj(from_file, to_file)
