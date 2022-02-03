import pickle
import string

import nltk
from nltk.corpus import stopwords
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB



def msg_transformation(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    emp = []
    for i in text:
        if i.isalnum():
            emp.append(i)
    text = emp[:]
    emp.clear()
    # Removing the stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            emp.append(i)
    text = emp[:]
    emp.clear()
    # Here we are stemming the text
    for i in text:
        ps = PorterStemmer()
        emp.append(ps.stem(i))

    return " ".join(emp)

tfidf = pickle.load(open('vectorizers.pkl' , 'rb'))
model = pickle.load(open('ml_model.pkl' , 'rb'))


st.title('Email/SMS Spam Classifier')
input_sms = st.text_area('Enter the message')
if st.button('Predict'):

# Transforming text
    transformed_msg = msg_transformation(input_sms)

# Vectorizing text
    vectorized_input = tfidf.transform([transformed_msg])

# Prediction
    result = model.predict(vectorized_input)[0]
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
