import urllib.request 
from bs4 import BeautifulSoup
import nltk
  
#industry=input("INDUSTRY REQUIRED:- ")
#relative_industry = input("Industry you want to juxtapose with: ")
# providing url

def para_extraction_from_wikipedia_using_key_words(industry):
    # getting all the paragraphs and putting them into a string of text
    url = "https://en.wikipedia.org/wiki/"+industry
    print(url)
    # opening the url for reading
    html = urllib.request.urlopen(url)
    
    # parsing the html file
    htmlParse = BeautifulSoup(html, 'html.parser')
    
    text=""" """
    for para in htmlParse.find_all("p"):
        text=text+para.get_text()

    return text

def para_extraction_from_any_website_given_url(article_url):
    # getting all the paragraphs and putting them into a string of text
    url = article_url
    print(url)
    # opening the url for reading
    html = urllib.request.urlopen(url)
    
    # parsing the html file
    htmlParse = BeautifulSoup(html, 'html.parser')
    
    text=""" """
    for para in htmlParse.find_all("p"):
        text=text+para.get_text()

    return text
    

#text = para_extraction_from_wikipedia_using_key_words(industry)
#text_juxtapose = para_extraction_from_wikipedia_using_key_words(relative_industry)

#print(text+"--------"+text)
#print(text+"--------"+text_juxtapose)

#sentence_split= nltk.tokenize.sent_tokenize(text)

#sentence_split_juxtapose= nltk.tokenize.sent_tokenize(text_juxtapose)
