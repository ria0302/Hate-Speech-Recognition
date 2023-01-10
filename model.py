import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from collections import Counter

# loading the data
dataset = pd.read_csv(r"C:/Machine Learning/Projects/labeled_data.csv/labeled_data.csv")
print(dataset.columns)
print(dataset['class'])

# analysing the data
# making pie chart
plt.pie(dataset["class"].value_counts().values, labels=dataset["class"].value_counts().index, startangle=90, autopct='%1.1f%%')
plt.show()
# data is highly unbalanced (1 has the greatest number of samples)

# balancing and reducing the data
class0 = dataset[dataset['class'] == 0]
class1 = dataset[dataset['class'] == 1].sample(n=10000)
class2 = dataset[dataset['class'] == 2]
# reduce the samples of class 1 and increase the samples of class 0 and 2 via undersampling and oversampling
new_dataset = pd.concat([class0, class1, class0, class2, class0, class0, class2])
print(new_dataset.shape)
plt.pie(new_dataset["class"].value_counts().values, labels=new_dataset["class"].value_counts().index, startangle=90, autopct='%1.1f%%')
plt.show()
# filtering by removing punctuations
# changing uppercase to lowercase data
# removing stopwords and lemmatizing
lemmatize = WordNetLemmatizer()


def process_data(x):

    x = x.translate(str.maketrans('', '', string.punctuation))
    x = x.lower()
    tokens = word_tokenize(x)
    del tokens[0]
    stop_words = stopwords.words('english')
    # create a dictionary of stopwords to decrease the find time from linear to constant
    stopwords_dict = Counter(stop_words)
    stop_words_lemmatize = [lemmatize.lemmatize(word) for word in tokens if word not in stopwords_dict]
    x_without_sw = (" ").join(stop_words_lemmatize)
    return x_without_sw


# update data
new_dataset["tweet"] = new_dataset["tweet"].apply(lambda x: process_data(x))

# as the data contains words -> mapping it to numbers
tfidf = TfidfVectorizer(max_features=10000)
transformed_vector = tfidf.fit_transform(new_dataset['tweet'])
print(new_dataset['tweet'].head())
print(transformed_vector.shape)
# the final input feature is transformed_vector

# splitting the data into test and train
X = transformed_vector
X_train, X_test, y_train, y_test = train_test_split(X, new_dataset['class'], random_state=42)

# training model
model = SVC(degree=4, C=1)
model.fit(X_train, y_train)
print("Train score SVM: {}".format(model.score(X_train, y_train)))
print("Test score SVM: {}".format(model.score(X_test, y_test)))
