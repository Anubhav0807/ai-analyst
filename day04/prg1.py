from sklearn.feature_extraction.text import CountVectorizer

text = ["free offer", "win money", "free money"]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(text)

print(vectorizer.get_feature_names_out())
print(x.toarray())
