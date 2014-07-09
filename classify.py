from __future__ import division
import time
import libsvm
import argparse
from cPickle import load
from learn import extractSift, computeHistograms, writeHistogramsToFile
from os import walk
import shutilke
import os

HISTOGRAMS_FILE = 'testdata.svm'
CODEBOOK_FILE = 'codebook.file'
MODEL_FILE = 'trainingdata.svm.model'

#to get and parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='classify images with a visual bag of words model')
    parser.add_argument('-c', help='path to the codebook file', required=False, default=CODEBOOK_FILE)
    parser.add_argument('-m', help='path to the model  file', required=False, default=MODEL_FILE)
    #parser.add_argument('input_images', help='images to classify', nargs='+')
    args = parser.parse_args()
    return args

input_images = []
for (dirpath, dirnames, filenames) in walk('testdata/'):
    input_images.extend(filenames)

for index,i in enumerate(input_images):
  filenames[index] = 'testdata' + '/' + filenames[index]


print "---------------------"
print "extract Sift features"
all_files = []
all_files_labels = {}
all_features = {}

args = parse_arguments()
model_file = args.m
codebook_file = args.c
fnames = filenames

#extract SIFT features for the testdata for classification
all_features = extractSift(fnames)


for i in fnames:
    all_files_labels[i] = 0  # initial label assignment

print "---------------------"
print "loading codebook from " + codebook_file
with open(codebook_file, 'rb') as f:
    codebook = load(f)

print "---------------------"
print "computing visual word histograms"
hist_start = time.time()
all_word_histgrams = {}
for imagefname in all_features:
    word_histgram = computeHistograms(codebook, all_features[imagefname])
    all_word_histgrams[imagefname] = word_histgram
hist_end = time.time()
print "time taken for histogram computation ",hist_end-hist_start


print "---------------------"
print "write the histograms to file to pass it to the svm"
histwrite_start=time.time()
nclusters = codebook.shape[0]
writeHistogramsToFile(nclusters,
                      all_files_labels,
                      fnames,
                      all_word_histgrams,
                      HISTOGRAMS_FILE)

histwrite_end = time.time()
print "time taken for writing histograms to file: ",histwrite_end-histwrite_start
print "---------------------"
print "Test Data Classification with SVM"

#print libsvm.test(HISTOGRAMS_FILE, model_file)

#classifier accuracy
kanclass = 0
engclass = 0
kantotal = 0
engtotal = 0

#print filenames

'''for i in filenames:
  if i.split('.')[0].endswith('0'):
    kantotal += 1
  else:
    engtotal += 1'''

#src = 'testdata/'
dest1 = 'kannada1/'
dest2 = 'english1/'

def copyFile(src, dest):
    try:
        shutil.copy(src, dest)
    
    except shutil.Error as e:
        parserrint('Error: %s' % e)
    
    except IOError as e:
        print('Error: %s' % e.strerror)

kanmisclass = []
engmisclass = []

count = 0
for index,i in enumerate(libsvm.test(HISTOGRAMS_FILE,model_file)):
  #print 'File:',filenames[index]
  count += 1
  print count
  if i == 0 :
    copyFile(filenames[index],dest1)
  else:
    print 'copied file',filenames[index]
    copyFile(filenames[index],dest2)
  
  if filenames[index].split('.')[0].endswith('0') and i == 0:
    kanclass += 1
  if not filenames[index].split('.')[0].endswith('0') and i == 1:
    engclass += 1

  if (filenames[index].split('.')[0].endswith('0') and i == 1):
    kanmisclass.append(filenames[index])

  if not filenames[index].split('.')[0].endswith('0') and i == 0:
    engmisclass.append(filenames[index])
  
'''
print 'No of Kannada Text Images',kantotal
print 'No of English Text Images',engtotal

print 'Classifier Accuracy:'

print 'Kannada match accuracy',float(kanclass/kantotal)*100
print  'English match accuracy',float(engclass/engtotal)*100


print 'Misclassified words'
print 'Kannada'
for i in kanmisclass:
  print i
print 'English'
for i in engmisclass:
  print i
'''