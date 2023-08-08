import csv
from html import entities
import urllib.request 
from bs4 import BeautifulSoup
import Google_results_scrape
import csv
import spacy

nlp=spacy.load("en_core_web_sm")
nlp2 = spacy.load("en_core_web_md")

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



def get_entities(Article):
   
   doc = nlp(Article)
   entities_of_article = list(doc.ents)
   return entities_of_article


def csv_file_creation(links,topic):

   with open(topic+'.csv','w')as f:
      thewriter = csv.writer(f)

      thewriter.writerow(['URL',"Text_data","entities"])

      for url in links:
         try:
            paragraph=para_extraction_from_any_website_given_url(url)
            #paragraph=summarize_text(paragraph)
            entities = get_entities(paragraph)
            required_row=[url,paragraph,entities]
            thewriter.writerow(required_row)
         except Exception as e:
            
            required_row=[url,"Page not allowing scraping"]
            thewriter.writerow(required_row)
         
def extract_data_from_google_search_and_store_data_in_csv():
      topic_of_search = input("Enter the topic of interest:- ")
      links = Google_results_scrape.scrape_google(topic_of_search)
      csv_file_creation(links,topic_of_search)

extract_data_from_google_search_and_store_data_in_csv()



