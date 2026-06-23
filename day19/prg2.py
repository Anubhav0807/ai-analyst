# import nltk
# nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
text = "I am learning Natural language Processing"
words = word_tokenize(text)

filtered_words = [
  word for word in words
  if word.lower() not in stopwords.words("english")
]

print(filtered_words)
