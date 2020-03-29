import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# nltk.download()
# nltk.download('popular')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()



import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import time

with open("intents.json") as file:
    data = json.load(file)

# print(data["intents"])

# already processed json data
try:
    # uncomment! fail this try to get updated data!
    load_data 
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            temp = nltk.word_tokenize(pattern)
            words.extend(temp)
            docs_x.append(temp)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []

        temp = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in temp:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# don't retrain the model if it has already been trained
try:
    train
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return numpy.array(bag)

def chat():
    print("Start talking with the bot (type q to quit)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit" or inp.lower() == "q" or inp.lower() == "exit":
            farewells = ["Goodbye!", "cya later!", "Farewell!", "See you!"]
            print(random.choice(farewells))
            time.sleep(2)
            break
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        # print(results)
        if results[0][results_index] > 0.6: # confidence level
            for search_tag in data["intents"]:
                if search_tag["tag"] == tag:
                    responses = search_tag["responses"]
                    print(random.choice(responses))
                    break
        else:
            error_messages = ["Sorry, I didn't get that, please try again.", "I don't understand your question. Please try again."]
            print(random.choice(error_messages))

chat()