#to delete .sift extension files

from os import walk

import os


files = []
for (dirpath, dirnames, filenames) in walk('../binaryclassifier/testdata/'):
    files.extend(filenames)

print files

for i in filenames:
	if i.endswith('.sift'):
		os.remove('../binaryclassifier/testdata/'+i)


