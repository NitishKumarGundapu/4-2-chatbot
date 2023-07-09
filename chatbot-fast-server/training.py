import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
import torch as t


lemmatizer = WordNetLemmatizer()
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))


pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))


training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
random.shuffle(training)

train_x,train_y = [],[]

for a in training:
    train_x.append(a[0])
    train_y.append(a[1])

train_x = t.tensor(train_x)
train_y = t.tensor(train_y)


class NeuralNet(t.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = t.nn.Linear(input_size, hidden_size)
        self.relu = t.nn.ReLU()
        self.l2 = t.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


input_size = len(train_x)
hidden_size = 8
num_classes = len(train_y)

model = NeuralNet(input_size, hidden_size, num_classes)
model = t.nn.Sequential(
    t.nn.Linear(len(train_x), len(train_y)),
    t.nn.ReLU(),
    t.nn.RNN(len(train_y),2),
    t.nn.ReLU(),
    t.nn.Linear(len(train_y), len(train_y)),
    t.nn.Softmax()
)

loss_fn = t.nn.MSELoss(reduction='sum')
optimizer = t.optim.Adam(model.parameters(), lr=1e-4)

for t in range(100):
    y_pred = model(train_x)

    loss = loss_fn(y_pred, train_y)
    if t % 10 == 9:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# model = Sequential()
# model.add(layers.Embedding(input_dim=len(train_x), output_dim=len(train_y)))
# model.add(layers.SimpleRNN(len(train_y)))
# model.add(Dense(len(train_y[0]), activation='softmax'))


# model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

# hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
# model.save('models/rnn_model.h5', hist)

print("model created")

print(model.summary())