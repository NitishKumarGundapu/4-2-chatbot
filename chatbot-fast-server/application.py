# import gradio as gr
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import nltk
import torch as t
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('pros_cons')
nltk.download('reuters')

import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


class NeuralNet(t.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = t.nn.Linear(input_size, hidden_size)
        self.l2 = t.nn.RNN(hidden_size, hidden_size)
        self.l3 = t.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out,hid = self.l2(out)
        out = self.l3(out)
        out = t.nn.functional.softmax(out)
        return out


def predict_class(sentence):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    p = np.array(p)
    p = t.from_numpy(p)
    p = p.view(1, -1).float()

    model = NeuralNet(37,16,4)
    model.load_state_dict(t.load('model/rnn_model.pth'))
    model.eval()

    res = model(p)

    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res[0]) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg)
    res = getResponse(ints, intents)
    return res

# gr.Interface(fn=chatbot_response, inputs=["text"], outputs=["text"],examples=[['hello'],['hi'],['help'],['bye']]).launch()\