import spacy

nlp=spacy.blank("en")

ruler=nlp.add_pipe('entity_ruler')

ruler.analyze_pipes()