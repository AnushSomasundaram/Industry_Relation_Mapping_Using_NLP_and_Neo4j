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

conn = Neo4jConnection(uri="bolt://localhost:7687", user="user1", pwd="pass")
Class = conn.query("Match(n:Class) return n.name")
description=conn.query("Match(n:Class) return n.Description")

for element in Class:

    name=str(element)
    #print(name[16:-2])
    name_to_access = name[15:-1]
    print(name)
    #print("MATCH (n:Subclass) \n where n.name="+name+ """\n SET n.data = \"Taylor\" \n Return n""" )
    #conn.query("MATCH (n:Class) \n where n.name="+name+ """\n SET n.keywords = \"Taylor\" \n Return n""" )
  
    










