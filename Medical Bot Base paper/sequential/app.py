import nltk
from flask import Flask, render_template, request
import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer 
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load necessary files and models
lemmatizer = WordNetLemmatizer() 
with open('Medical Bot/intents.json') as json_file:
    intents = json.load(json_file)

words = pickle.load(open('Medical Bot\words2.pkl','rb'))
classes = pickle.load(open('Medical Bot\classes2.pkl','rb'))
model = load_model('Medical Bot\chatbotmodel2.h5')

def clean_the_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow_of_sentence(sentence):
    sentence_words = clean_the_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_the_class(sentence):
    bow = bow_of_sentence(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_the_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    message = request.form['message']
    response = "I am sorry, but I cannot answer that."
    ints = predict_the_class(message)
    res = get_the_response(ints, intents)
    response = res
    return response

if __name__ == '__main__':
    app.run(debug=True)
