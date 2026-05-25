from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Better dataset
text = [
  # Spam messages
  "Congratulations you won a free lottery ticket",
  "Claim your free cash prize now",
  "Win money instantly by clicking this link",
  "Exclusive offer just for you buy now",
  "Get rich quickly with this amazing opportunity",
  "Free vacation package waiting for you",

  # Not spam
  "Team meeting scheduled for tomorrow morning",
  "Please review the project report",
  "Lunch discussion with client today",
  "Can you send the presentation slides",
  "Project deadline has been extended",
  "Let us discuss the quarterly budget"
]

spam_label = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

# Convert text into numeric vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(text)

# Train model
model = MultinomialNB()
model.fit(x, spam_label)

# Test messages
test_messages = [
  "Win a free vacation now",
  "Project meeting tomorrow",
  "Claim your money prize",
  "Please send the budget report"
]

test = vectorizer.transform(test_messages)

predictions = model.predict(test)

# Show results
for msg, pred in zip(test_messages, predictions):
  print(f"Message: '{msg}'")
  print("Spam" if pred == 1 else "Not Spam")
  print()