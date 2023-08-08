
from html import entities
from multiprocessing.sharedctypes import Value
import urllib.request 
from bs4 import BeautifulSoup
from matplotlib.pyplot import get
from regex import E
import Google_results_scrape
import requests
import redis
import spacy

redis_file = redis.Redis(host="localhost",port=6379)
nlp=spacy.load("en_core_web_sm")

def start_to_end(url):
    
      print(url)
      # opening the url for reading
      html = urllib.request.urlopen(url)
      
      # parsing the html file
      htmlParse = BeautifulSoup(html, 'html.parser')
      
      text=""" """
      for para in htmlParse.find_all("p"):
            text=text+para.get_text()

      data_from_url={"url":url,"Text_Data": text}

      redis_file.hmset(url,data_from_url)
      

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


def data_put_into_dictionaries(url):
      try:
            paragraph=str(para_extraction_from_any_website_given_url(url))
            data_from_url={"url":url,"Text_Data": paragraph}
            #redis_file.hmset(url,data_from_url)
            return data_from_url
            
      except Exception as e:
            data_from_url={"URL": url ,"Text_data":"Page not allowing scraping"}
            #redis_file.hmset(url,data_from_url)
            return data_from_url
            

def data_upload_with_redis_job_scheduling(url):
      dictionary_info=dict(data_put_into_dictionaries(url))
      redis_file.hmset(url,dictionary_info)
      

def get_entities(Article):
   
   doc = nlp(Article)
   entities_of_article = str(list(doc.ents))
   return entities_of_article

def Entity_Upload_with_redis_job_scheduling(key):
      
      Article=str(redis_file.hget("Text_data",key))
      entities=get_entities(Article)
      dictionary_info = dict(redis_file.hgetall(key))
      dictionary_info["Entities"]= entities
      redis_file.delete(key)
      redis_file.hmset(key,dictionary_info)


def redis_complete_text_data_upload(links):
      for url in links:
            data_upload_with_redis_job_scheduling(url)
            
                        
         
def extract_data_from_google_search_and_store_data_in_redis():
      topic_of_search = input("Enter the topic of interest:- ")
      links = Google_results_scrape.scrape_google(topic_of_search)
      redis_complete_text_data_upload(links,topic_of_search)





#data_upload_with_redis_job_scheduling("https://en.wikipedia.org/wiki/History_of_Wikipedia")