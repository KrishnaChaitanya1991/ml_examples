# spam classifier using naive bayes
import os
import io
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
spam_emails_path = "./emails/spam"
ham_emails_path = "./emails/ham"

def read_files(path):
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            path = os.path.join(root, file_name)
            f = open(path, 'r', encoding='latin1')
            message = f.read()
            f.close()
            yield path, message


def create_data_frame(path, classification):
    rows = []
    index = []
    for file_path, message in read_files(path):
        rows.append({'message': message, 'class': classification})
        index.append(file_path)
    return pd.DataFrame(rows, index=index)

data = pd.DataFrame({'message': [], 'class':[]})
data = data.append(create_data_frame(spam_emails_path, 'spam'))
data = data.append(create_data_frame(ham_emails_path, 'ham'))
# print(data.head())

vectorizer = CountVectorizer()
word_vector =vectorizer.fit_transform(data['message'].values)
print(word_vector.shape)

nb_classifier = MultinomialNB()
nb_classifier.fit(word_vector, data['class'].values)

examples = ['Free Viagra', 'Hey you won 100000000 $ deposit 5 $ and get it!!!!', 'You are awesome in this epoch']
predictions = nb_classifier.predict(vectorizer.transform(examples))
print(predictions)