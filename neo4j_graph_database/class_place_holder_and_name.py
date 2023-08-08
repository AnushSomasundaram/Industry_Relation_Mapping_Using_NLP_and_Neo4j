import csv
import pandas as pd
import re

df_nic=pd.read_csv("/Users/somradhakrishnan/Desktop/Taatatsu/Project0/code/neo4j_graph_database/NIC.csv")

class_place_holder_to_name=[]
i=0
set_section=""
sum=0

f = open('Project0/code/neo4j_graph_database/Class_place_holder_name.csv', 'w')
writer = csv.writer(f)
row_1=["Placeholder_of_class","Name"]
writer.writerow(row_1)

while i<len(df_nic):
    
    i=i+1
    if re.search("^[0-9]+$", str(df_nic.loc[i,"Group"])) :
        temp_for_section ="Group_"+str(df_nic.loc[i,"Group"])
        set_section= temp_for_section 
        #print(set_section)
    
    description="""  """
        
  
        
    if re.search("^[0-9]+$", str(df_nic.loc[i,"Class"])) :
        temp_for_class =str(df_nic.loc[i,"Class"])
        
        j=i
        while str(df_nic.loc[j,"Sub-class"])=="nan" :
            #print(df.loc[j,"Description"])
            description = description+" " + str(df_nic.loc[j,"Description"])
            #print(description)
            j=j+1
        
        sum=sum+1
        temp=temp_for_class
        string=str(df_nic.loc[i,"Description"])
        #class_place_holder_to_name.append(set_section)
        #class_place_holder_to_name.append(description)
        temp=[temp_for_class,str(df_nic.loc[i,"Description"])]
        writer.writerow(temp)
        
        