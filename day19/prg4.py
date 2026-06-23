import spacy
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

text = "Amazon, the e-commerce giant, is headquatered in Seattle, Washington." + \
  " The company was founded by Jeff Bezos in 1994."
doc = nlp(text)

for ent in doc.ents:
  print(ent.text, ent.label_) # GPE - Geopolitical Entity
