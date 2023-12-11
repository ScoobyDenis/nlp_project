import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle

# Устанавливаем стоп-слова и лемматизатор
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Загружаем данные для обучения
df = pd.read_csv('/home/scooby/Desktop/datasets/IMDB.csv')
positive_reviews = df[df['sentiment'] == 'positive']  # Положительные отзывы
negative_reviews = df[df['sentiment'] == 'negative']  # Отрицательные отзывы

# Предобрабатываем и очищаем данные
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Создаем тренировочный набор данных
train_data = [(preprocess(text), 'positive') for text in positive_reviews] + [(preprocess(text), 'negative') for text in negative_reviews]

# Создаем объект для векторизации текста
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform([text for text, label in train_data])

# Обучаем модель
classifier = LinearSVC()
classifier.fit(train_features, [label for text, label in train_data])

# Сохраняем модель на диск
pickle.dump((vectorizer, classifier), open("model.pkl", "wb"))