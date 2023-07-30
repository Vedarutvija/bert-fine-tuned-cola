import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import pandas as pd
import re
import spacy
import wikipediaapi
import openai
openai.api_key = "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"
# Load the dataset
df = pd.read_csv('Position_Salaries.csv')
# Preprocess the data
df.columns = [re.sub(r'[^\w\s]', '', col.lower()) for col in df.columns]
df = df.applymap(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))

# Define a function to extract keywords from user input
def extract_keywords(sentence):
    # Preprocess user input
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())

    # Extract keywords related to column names
    col_keywords = []
    for col in df.columns:
        if col in sentence:
            col_keywords.append(col)

    # Extract keywords related to row values
    row_keywords = []
    for index, row in df.iterrows():
        if row['position'] in sentence:
            row_keywords.append(row['position'])

    # Return list of keywords
    return col_keywords, row_keywords


# Define a function to query the dataset and return the corresponding answer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(sentence):
    sent = tokenize(sentence)
    X = bag_of_words(sent, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    result =""
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                result = random.choice(intent['responses'])

        return result
    else:
        while True:
            model_engine = "text-davinci-003"
            completion = openai.Completion.create(
                engine=model_engine,
                prompt=sentence,
                max_tokens=1030,
                n=1,
                stop=None,
                temperature=0.5,
        )
            response = completion.choices[0].text
            return response
        # subject = ""
        # for token in doc:
        #     if token.dep_ == "nsubj" and token.pos_ == "NOUN":
        #         subject = token.text
        #         break
        # if subject != "":
        #     summary = get_wiki_summary(subject)
        #     return summary

def process_input(sentence):
    col_keywords, row_keywords = extract_keywords(sentence)
    if len(col_keywords) > 0 and len(row_keywords) > 0:
        for row in row_keywords:
            for col in col_keywords:
                value = df.loc[df['position'] == row, col]
                if not value.empty:
                    if isinstance(value.values[0], str):
                        return value.values[0]
                    elif isinstance(value.values[0], int):
                        return f"{value.values[0]:,}"
                    elif isinstance(value.values[0], float):
                        return f"{value.values[0]:.2f}"
                    else:
                        return str(value.values[0])
        return "Sorry, I couldn't find any matching values in the dataset."
    else:
        ints = get_response(sentence)
        return ints

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = process_input(sentence)
        print(resp)

