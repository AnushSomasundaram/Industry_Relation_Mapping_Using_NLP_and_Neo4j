#Program to find out tf-idf values of words.
#We import word_tokenize from the nltk module to tokenize the sentences into words
from nltk.tokenize import word_tokenize

#We import numpy to give us a list of zeros
import numpy as np


#We import scrape to get the dataset from wikipedia 
import scrape

#Creating a list for the sentences and word_set

sentences=[]
sentences_juxtapose=[]
word_set=[]

text=scrape.sentence_split

#to get words to compare percentage of similarity
text_justapose=scrape.sentence_split_juxtapose

#code snippet to split into sentences and words

for sent in text :

   x=[i.lower() for i in word_tokenize(sent) if i.isalpha()]
   sentences.append(x)
   

for sent in text_justapose :
   
   x=[i.lower() for i in word_tokenize(sent) if i.isalpha()]
   sentences_juxtapose.append(x)
   for word in x:
   
      if word not in word_set:
         word_set.append(word)


word_set = set(word_set)

# A document here is nothing but a scentence we call it a document as the variable names can't have any conflict. 

total_documents=len(sentences)



#Create an index for each word in the vocabulary of the data set.
index_dict = { } #Dictionary to store index for each word
i=0

for word in word_set:
   index_dict[word]=i
   i+=1

# The following function is to create a dictionary to store the number of sentences a given word is in
def count_dict(sentences):
   word_count={}
   word_count[word]=0
   for sent in sentences:
      if word in sent:
         word_count[word] += 1

   return word_count

word_count=count_dict(sentences)

#Term Frequency :- number of repetions of the words in sentence/words in sentence
def termfrequency(document,word):
   N= len(document)
   occurance = len([token for token in document if token==word]) 
   #print(word)
   return occurance/N  #----->Value of term frequency generated here

#inverse document frequency :- number of scentences that have the required word in them/ total number of words.
def inverse_doc_frequency(word):
   try:
      word_occurance = word_count[word]+1
   
   except:
      word_occurance=1
   
   return np.log(total_documents/word_occurance) #----->Value of inverse document frequency generated here

#Now to calculate tf-idf , tf-idf :- (tf*idf) of each token

def tf_idf(sentence):
   tf_idf_vec = np.zeros((len(word_set),))
   for word in sentence:
      tf = termfrequency(sentence,word)
      idf=inverse_doc_frequency(word)
      value=tf*idf   #-----> This is the final value for each word
      tf_idf_vec[index_dict[word]] = value

   return tf_idf_vec


#TF-IDF encoded text corpus

vectors =[]
final_sent=[]

#list of a list of values
for sent in sentences:
   vec=tf_idf(sent)
   final_sent.append(sent)
   vectors.append(vec)


#[print(i, end=' ') for i in vectors[0]]


z=0
while z<len(vectors[0]) :
   if vectors[0][z] != 0.0:
      print(word_set[z])
      print(vectors[0][z])
   z=z+1


      








