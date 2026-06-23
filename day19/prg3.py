from textblob import TextBlob

for text in ["I love learning NLP", "I hate learning NLP"]:
  analysis = TextBlob(text)
  polarity = analysis.sentiment.polarity
  print("Sentence:", text)
  print("Polarity:", polarity)
  print("Sentiment:", end=' ')

  if polarity > 0:
    print("Positive")
  elif polarity < 0:
    print("Negative")
  else:
    print("Neutral")
