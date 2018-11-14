import pandas as pd
import csv
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.utils import to_categorical

LSTM_Model_Name = 'LSTM_Model'

# Method to convert TXT to CSV
def convert_to_csv(input_txt):
    output_csv = 'ner_test_data.csv'
    stripped = []
    lines = []
    number = 0
    with open(input_txt, 'r') as in_file:
        for line in in_file:
            if 0 == len(line.strip()):
                number += 1
                continue
            parts = line.strip().split(' ')
            stripped.append(str(number) + ',' + parts[0] + ',' + parts[1] + ',' + parts[2] + ',' + parts[3])

        # stripped = (line.strip() )
        for line in stripped:
            if line:
                # t = (line.replace('\t',','))
                line = list(line.split(","))
                lines.append(tuple(line))

        with open(output_csv, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('Sentence_Number', 'Word', 'POS', 'Tag','Label'))
            writer.writerows(lines)

    return output_csv

# Problem with working with new data.
def predict_on_test_set(filepath):
    df = pd.read_csv(filepath, sep=',', error_bad_lines=False)

    print(df.head())
    test_sentences = getSentences(df)

    print(test_sentences)

    # Get Unique Words in the dataset.
    words = list(set(df["Word"].values))
    words.append("PADDING")

    # Number of words
    number_of_words = len(words)

    # Get Unique Labels in the dataset.
    labels = list(set(df["Label"].values))

    # Number of labels
    number_of_lables = len(labels)

    # Set maximum sentence length. If sentence length is less than max_sentence_length, add padding to the end.
    max_sentence_length = 50

    # Convert words into index values.
    word2idx = {wordIdx: wordIndex for wordIndex, wordIdx in enumerate(words)}

    # Read all the sentences into X array.
    X = [[word2idx[w[0]] for w in s] for s in test_sentences]

    # Pad sentences using pad_sequences method from Keras. Padding done with string PADDING
    X_test = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=number_of_words - 1)

    lstm = joblib.load(LSTM_Model_Name)

    for test_sentence in X_test:
        p = lstm.predict(np.array([test_sentence]))
        p = np.argmax(p, axis=-1)
        for w, pred in zip(test_sentence, p[0]):
            print(words[w], labels[pred])
        print('\n')

# Get sentences from data frame (CSV file)
def getSentences(df):
    agg_func = lambda s: [(word, pos, sp_tag, label) for word, pos, sp_tag, label in
                          zip(s["Word"].values.tolist(),s["POS"].values.tolist(),
                          s["Tag"].values.tolist(),s["Label"].values.tolist())]

    grouped = df.groupby("Sentence_Number").apply(agg_func)
    sentences = [s for s in grouped]

    return sentences

# Training and Testing a model
def train():
    input_txt = 'ner_dataset.txt'
    csv_file = convert_to_csv(input_txt)
    csv_file = 'lstm_ner_dataset.csv'

    # Read csv file
    df = pd.read_csv(csv_file, sep=',', error_bad_lines=False)

    # Get Unique Words in the dataset.
    words = list(set(df["Word"].values))
    words.append("PADDING")

    # Number of words
    number_of_words = len(words)

    # Get Unique Labels in the dataset.
    labels = list(set(df["Label"].values))

    # Number of labels
    number_of_lables = len(labels)

    sentences = getSentences(df)

    # Set maximum sentence length. If sentence length is less than max_sentence_length, add padding to the end.
    max_sentence_length = 50

    # Convert words into index values.
    word2idx = {wordIdx: wordIndex for wordIndex, wordIdx in enumerate(words)}

    # Convert labels into index values.
    label2idx = {labelIdx: labelIndex for labelIndex, labelIdx in enumerate(labels)}

    print(word2idx["Peter"])

    # Read all the sentences into X array.
    X = [[word2idx[w[0]] for w in s] for s in sentences]

    # Pad sentences using pad_sequences method from Keras. Padding done with string PADDING
    X = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=number_of_words - 1)

    # Read all the labels into X array.
    y = [[label2idx[w[3]] for w in s] for s in sentences]

    # Pad sentences using pad_sequences method from Keras. Padding done with String O.
    y = pad_sequences(maxlen=max_sentence_length, sequences=y, padding="post", value=label2idx["O"])

    # Convert Lables into Categorical values
    y = [to_categorical(i, num_classes=number_of_lables) for i in y]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    print('Number of Train Samples : ', len(X_train))
    print('Number of Test Samples : ', len(X_test))

    # Keras model creation
    input = Input(shape=(max_sentence_length,))
    model = Embedding(input_dim=number_of_words, output_dim=50, input_length=max_sentence_length)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(number_of_lables, activation="softmax"))(model)  # softmax output layer

    model = Model(input, out)

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, np.array(y_train), batch_size=32, epochs=2, validation_split=0.1, verbose=1)

    LSTM_Model_Name = 'LSTM_Model'

    # Create a CRF model file
    joblib.dump(model, LSTM_Model_Name)

    lstm = joblib.load(LSTM_Model_Name)  # Load model

    for test_sentence in X_test:
        p = lstm.predict(np.array([test_sentence]))
        p = np.argmax(p, axis=-1)
        for w, pred in zip(test_sentence, p[0]):
            print(words[w], labels[pred])
        print('\n')

if __name__ == '__main__':
    train()
    # lstm_test_data = 'lstm_ner_test_data.csv'
    # predict_on_test_set(lstm_test_data)