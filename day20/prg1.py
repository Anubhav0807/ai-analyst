import nltk
import spacy
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger_eng')

text = "Amazon is headquartered in Seattle. Jeff Bezos founded Amazon and I love this company."
print("Original text:", text)

tokens = word_tokenize(text)
print("\nTokens:", tokens)

stop_words = set(stopwords.words("english"))
filtered_words = [word for word in tokens if word.lower() not in stop_words]
print("\nStopwords:", filtered_words)

stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in filtered_words]
print("\nStemming:", stems)

lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word) for word in filtered_words]
print("\nLemmatization:", lemmas)

pos_tags = nltk.pos_tag(tokens)
print("\nPOS Tags:", pos_tags)

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print("\nNER:")
for ent in doc.ents:
  print(ent.text, "-", ent.label_)

analysis = TextBlob(text)
polarity = analysis.sentiment.polarity
print("\nSentiments:")
print("Polarity:", polarity)
if polarity > 0:
  print("Positive")
elif polarity < 0:
  print("Negative")
else:
  print("Neutral")
