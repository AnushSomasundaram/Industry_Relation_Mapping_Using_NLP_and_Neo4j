from neo4j import __version__ as neo4j_version
from neo4j import GraphDatabase
from regex import F
from sqlalchemy import desc
import csv
#import spacy
from rake_nltk import Rake
class Neo4jConnection:
    
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response
"""
nlp = spacy.load("en_core_web_sm")
def get_entities(Article):
   
   doc = nlp(Article)
   entities_of_article = doc.ents
   for ent in entities_of_article:
       print(ent.text,ent.label_)
   return entities_of_article
"""

rake_nltk_var=Rake()
#Rapid Automatic Keyword Extraction
def get_keywords_using_rake_algorithm_from_nltk(text):
    rake_nltk_var.extract_keywords_from_text(text)
    keyword_extracted = rake_nltk_var.get_ranked_phrases()
    return keyword_extracted


conn = Neo4jConnection(uri="bolt://localhost:7687", user="USER1", pwd="password")



