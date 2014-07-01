
import urllib

def scrape_images():
  i = 0
  while (i<50):	
	wordno = 6008
	wordid = 'bin_'+repr(wordno)
	f = open(wordid+'.png','wb')
	f.write(urllib.urlopen('http://blr.tachyon.in:9000/images/1/sharath/tachyonocr/wordmatch/BookTesting/words/'+wordid).read())
	wordno += 1
	f.close()
	wordno += 1
	i += 1

url ="http://blr.tachyon.in:9000/images/1/sharath/tachyonocr/wordmatch/BookTesting/words/bin_6008.png"

#get the images
scrape_images()
