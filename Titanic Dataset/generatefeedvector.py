import feedparser, math
import re
import nltk
from nltk.stem.snowball import SnowballStemmer


def getwordcounts(filenm):
  """
	 Returns title and dictionary of word counts for the document
  """
  # Parse the feed
  try:
        doc = open(filenm, 'r').read()
  except:
        print("Error could not open file")
  wc={}
  words = getwords(doc)
  
  for word in words:
      wc.setdefault(word,0)
      wc[word]+=1

  with open(filenm) as f:
    return f.readline(),wc



def getwords(html):
  """
	Tokenizes the document
  """
  # Remove all the HTML tags
  txt=re.compile(r'<[^>]+>').sub('',html)

  # Split words by all non-alpha characters
  words=re.compile(r'[^A-Z^a-z]+').split(txt)

  # Removal of stopwords
  stopwords = nltk.corpus.stopwords.words('english')
  
  #usage of stemmming algo.
  stemmer = SnowballStemmer("english")
  stemmer.stem
  swords = [ stemmer.stem(wrd) for wrd in words]

  #print len(swords)
  return [word.lower() for word in swords if (word!='' and word not in stopwords )]


apcount={}
wordcounts={}
n = input("Please enter the number of documents to be vectorised : ")

for i in range(1,n+1):
  try:
    filenm = str(i)+ ".txt"
    print "Parsing file :",
    print filenm
   
    title,wc = getwordcounts(filenm)

    wordcounts[title]=wc
    for word,count in wc.items():
      apcount.setdefault(word,0)
      if count>1:
        apcount[word]+=1
  except:
    print 'Failed to parse feed %s.txt' % i

wordlist=[]
for w,bc in apcount.items():
  frac=float(bc)/n
  if frac>0.1 and frac<0.3:
    wordlist.append(w)
#print apcount.items()
df = {}

for word in wordlist:
  t = 0
  for blog,wc in wordcounts.items():
    if word in wc:
      t = t + 1
  df[word] = math.log((float(n) / t),10)



out = file('blogdata2.txt','w')
out.write('Blog')
for word in wordlist: out.write('\t%s' % word)
out.write('\n')
for blog,wc in wordcounts.items():
  print blog
  out.write(blog)
  for word in wordlist:
    if word in wc:
      temp = (wc[word]) 
      out.write('\t%f' % temp)
    else:
      out.write('\t0')
  out.write('\n')
x = input("Press any key to exit......")