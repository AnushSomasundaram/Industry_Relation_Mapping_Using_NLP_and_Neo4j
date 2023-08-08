#this prefix code is required to connect to a neo4j database..... and then the actaul connection is done using Neo4jConnection(....)
from neo4j import __version__ as neo4j_version
from neo4j import GraphDatabase
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

conn = Neo4jConnection(uri="bolt://localhost:7687", user="USER1", pwd="password")

#from here you can start typing queries

import pandas as pd
import re
import numpy as np
df = pd.read_csv('Project0/code/neo4j_graph_database/NIC.csv')
df = df.drop(["Unnamed: 4","Unnamed: 5","Unnamed: 6","Unnamed: 7","Unnamed: 8","Unnamed: 9","Unnamed: 10"],axis =1)


#The 
#conn.query("""Create(INDUSTRY:INDUSTRY {name:"Industry"})""")
conn.query("Match (N) DETACH DELETE N")
conn.query("""CREATE (INDUSTRIES_IN_INDIA:INDUSTRIES {name:"Industries In India"})""")


#section nodes
i=0
queries = """ """
while i<len(df):
    if str(df.loc[i,"Group"]).startswith("SECTION"):
        temp_for_section =str(df.loc[i,"Group"])
        queries = queries + ("\nCREATE("+ temp_for_section[0:7]+"_"+temp_for_section[8] + ":SECTION {name:'"+ temp_for_section[11:] +"'})-[:SECTION_OF_INDUSTRY]->(INDUSTRIES_IN_INDIA)-[:SECTION_OF_INDUSTRY]->("+temp_for_section[0:7]+"_"+temp_for_section[8]+")")
    i=i+1



#division nodes

i=0
set_section=""
while i<len(df):
    
    if str(df.loc[i,"Group"]).startswith("SECTION"):
        temp_for_section =str(df.loc[i,"Group"])
        set_section=temp_for_section[0:7]+"_"+temp_for_section[8] 
        
    if str(df.loc[i,"Group"]).startswith("DiviSion"):
        temp_for_division =str(df.loc[i,"Group"])
        queries = queries + ("\nCREATE("+temp_for_division[0:8]+temp_for_division[9:11] +":DiviSion {name:'"+ temp_for_division[13:] +"'})-[:Division_under_section]->(" + set_section + ")-[:Division_under_section]->("+ temp_for_division[0:8]+temp_for_division[9:11] + ")")
    i=i+1




i=0
set_section=""
while i<len(df):
    
    
    if str(df.loc[i,"Group"]).startswith("DiviSion"):
        temp_for_section =str(df.loc[i,"Group"])
        set_section= temp_for_section[0:8]+temp_for_section[9:11] 
        
    
    
    if re.search("^[0-9]+$", str(df.loc[i,"Group"])) :
        #print(df.loc[i,"Group"])
        temp_for_division =str(df.loc[i,"Group"])
        queries = queries + ("\nCREATE(Group_"+temp_for_division+":Group {name:'" + str(df.loc[i,"Description"]) +"'})-[:Group]->(" + set_section + ")-[:Group]->(Group_"+temp_for_division+")")
    
    i=i+1

i=0
set_section=""
sum=0

while i<len(df):
    
   try:
      if re.search("^[0-9]+$", str(df.loc[i,"Class"])) :
        temp_for_section ="Class_"+str(df.loc[i,"Class"])
        set_section= temp_for_section 
        #print(set_section)
    
      description="""  """
        
  
      
      if re.search("[0-9]+$", str(df.loc[i,"Sub-class"])) :
         temp_for_sub_class =str(df.loc[i,"Sub-class"])
         temp_for_sub_class=temp_for_sub_class[:-2]
         #print(temp_for_sub_class)
         
         j=i+1
         description=description = df.loc[i,"Description"]
         
         if str(df.loc[j,"Sub-class"])=="nan":
               while str(df.loc[j,"Sub-class"])=="nan" :
                  #print(df.loc[j,"Description"])
                  description = description+" "+ str(df.loc[j,"Description"])
                  #print(description)
                  j=j+1
            
               
         
         #print(description)
         queries = queries + ("\nCREATE(Subclass_"+temp_for_sub_class+":Subclass {name:'" + description+"'})-[:CLASS]->(" + set_section + ")-[:CLASS]->(Subclass_"+temp_for_sub_class+")")
      
         
         sum=sum+1
      i=i+1
   except:
      pass
      i=i+1

i=0
set_section=""
sum=0
while i<len(df):
    
    
    if re.search("^[0-9]+$", str(df.loc[i,"Group"])) :
        temp_for_section ="group_"+str(df.loc[i,"Group"])
        set_section= temp_for_section 
        #print(set_section)
    
    description="""  """
        
  
        
    if re.search("^[0-9]+$", str(df.loc[i,"Class"])) :
        temp_for_class =str(df.loc[i,"Class"])
        
        j=i
        while str(df.loc[j,"Sub-class"])=="nan" :
            #print(df.loc[j,"Description"])
            description = description+" " + str(df.loc[j,"Description"])
            #print(description)
            j=j+1
        
        sum=sum+1
        
        
        queries = queries + ("\nCREATE(Class_"+temp_for_class+":Class {name:'" + str(df.loc[i,"Description"])+"',Description :'"+description +"'})-[:Group]->(" + set_section + ")-[:Group]->(Class_"+temp_for_class+")")
    
    
    
    
    i=i+1

#print(queries)


conn.query(queries)
