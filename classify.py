import libsvm
import argparse
from cPickle import load
from learn import extractSift, computeHistograms, writeHistogramsToFile
from os import walk

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
for (dirpath, dirnames, filenames) in walk('../binaryclassifier/testdata/'):
    input_images.extend(filenames)

for index,i in enumerate(filenames):
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
all_features = extractSift(fnames)
for i in fnames:
    all_files_labels[i] = 0  # initial lael i

print "---------------------"
print "loading codebook from " + codebook_file
with open(codebook_file, 'rb') as f:
    codebook = load(f)

print "---------------------"
print "computing visual word histograms"
all_word_histgrams = {}
for imagefname in all_features:
    word_histgram = computeHistograms(codebook, all_features[imagefname])
    all_word_histgrams[imagefname] = word_histgram

print "---------------------"
print "write the histograms to file to pass it to the svm"
nclusters = codebook.shape[0]
writeHistogramsToFile(nclusters,
                      all_files_labels,
                      fnames,
                      all_word_histgrams,
                      HISTOGRAMS_FILE)

print "---------------------"
print "Test Data Classification with SVM"

#print libsvm.test(HISTOGRAMS_FILE, model_file)

for index,i in enumerate(libsvm.test(HISTOGRAMS_FILE,model_file)):
  print 'File:',filenames[index]
  print 'Classification:',i


