import spacy
import scrape


industry_text = scrape.text
relative_industry_text = scrape.text_juxtapose



nlp = spacy.load("en_core_web_md")

def compare_two_documents(Article1,Article2):
   # This function compares two documents and see what level of similarity they have
   
   doc1 = nlp(Article1)
   
   doc2 = nlp(Article2)
   
   document_similarity_value = doc1.similarity(doc2) 
   
   print("The two documents have a similarity value of :- ")
   print(document_similarity_value)

def get_entities(Article):
   
   doc = nlp(Article)
   entities_of_article = list(doc.ents)
   return entities_of_article

def entity_overlapping_set(Article1,Article2):

      entities_of_set_1=get_entities(Article1)
      print(entities_of_set_1)
      entities_of_set_2=get_entities(Article2)
      print(entities_of_set_2)
      i=0
      """
      print("All the similar entites are:- ")
      
      while i<len(entities_of_set_1):
         if entities_of_set_1[i]==entities_of_set_2[i]:
            print(entities_of_set_1[i])
            i=i+1
      """

#compare_two_documents(industry_text,relative_industry_text)

#compare_two_documents(scrape.para_extraction_from_any_website_given_url("https://anushsom.medium.com/greedy-algorithms-for-dummies-by-a-dummy/"),
#                        scrape.para_extraction_from_any_website_given_url("https://anushsom.medium.com/graphs-and-graph-traversal-algorithms-for-dummies-by-a-dummy-dd19149f4882"))

#print(get_entities(industry_text))

#entity_overlapping_set(industry_text,relative_industry_text)
