#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, jsonify
import joblib
import traceback
import pandas as pd
from sklearn.compose import ColumnTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re


app = Flask(__name__)

model, vect = joblib.load('./lr_l2_hugo.joblib')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def clean_text(text):    
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub("\'", "", text)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english')) 
    words = nltk.word_tokenize(text)
    text_filtered = " ".join([word for word in text.split() if word not in stop_words])
    return text_filtered

wordnet_lemmatizer = WordNetLemmatizer()

def lem(text):
    words = text.split()
    return " ".join([wordnet_lemmatizer.lemmatize(word) for word in words])

@app.route('/predict', methods=['POST'])
def predict():
    try:               
        title = request.form.get('title')
        plot = request.form.get('plot')
                
        input_data = pd.DataFrame({            
            'title':[title],
            'plot': [plot]})
                            
        X = input_data['plot']
        X = input_data['plot'].apply(clean_text)
        X = X.apply(remove_stopwords)
        X = X.apply(lambda y: lem(y))
        X_tfidf = vect.transform(X)
               
        prediction = model.predict_proba(X_tfidf)
        
        cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
        
        res = pd.DataFrame(prediction, columns=cols)       
        res = res.loc[:, (res != 0).all(axis=0)]

        return jsonify({f'Género película {title}': res.to_json(orient="columns")})
        
                
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


# In[ ]:




