import sys
import nltk
import sklearn_crfsuite
from gensim.models import Word2Vec
from sklearn_crfsuite import metrics
from sklearn.externals import joblib

# Choose any of the following models for
# model_name = 'chunking_crf_model_pos'
# model_name = 'chunking_crf_model_pos_next_previous_word'
model_name = 'chunking_crf_model_pos_next_previous_word_case_start_word'

# New Model file name if new features are chosen.
# Use train_model() method to train new model.
# new_model_name = 'chunking_crf_model'

# Word2vec model name for work embeddings
w2v_model_name = 'chunking_w2v_word_embedding_model'
w2v_model = ''

# Get sentence count.
def get_sentence_count(filepath):
    count = 0
    with open(filepath) as chunking_data:
        for line in chunking_data.readlines():
            if 0 == len(line.strip()):
                count += 1
    print (count)
    return count

# Method used to split the data set and
# get test and train split given filepath.
def get_test_train_split_from_file(filepath, tt_division=0.7):
    sentence = []
    train = []
    test = []
    count = 0
    sentence_count = get_sentence_count(filepath)
    sentence_count = sentence_count * tt_division

    with open(filepath) as chunking_data:
        for line in chunking_data.readlines():

            if 0 == len(line.strip()):
                if count <= sentence_count:
                    train.append(sentence)
                else:
                    test.append(sentence)
                count += 1
                sentence = []
                continue
            else:
                parts = line.split(' ')
                temp = (parts[0].strip(), parts[1].strip(), parts[2].strip())

            sentence.append(temp)
    return train, test

# Features - POS
def word2features_pos(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'postag': postag,
    }

    return features

# Features - POS
# Previous word Next Word POS tag
def word2features_pos_next_previous_word(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'postag': postag,
    }

    # Get previous word and pos tag
    if i > 0:
        previous_word = sent[i-1][0]
        previous_postag = sent[i-1][1]
        features.update({
            'previous_postag': previous_postag,
        })
    else:
        features['Start'] = True

    # Get next word and pos tag
    if i < len(sent)-1:
        next_word = sent[i+1][0]
        next_postag = sent[i+1][1]
        features.update({
            'next_postag': next_postag,
        })
    else:
        features['End'] = True

    return features

# Features - POS
# Previous word Next Word POS tag
# Previous and Next Word Case
def word2features_pos_next_previous_word_case(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'isupper': word[0].isupper(),
        'postag': postag,
    }

    # Get previous word and pos tag
    if i > 0:
        previous_word = sent[i-1][0]
        previous_postag = sent[i-1][1]
        features.update({
            'isupper': previous_word[0].isupper(),
            'previous_postag': previous_postag,
        })
    else:
        features['Start'] = True

    # Get next word and pos tag
    if i < len(sent)-1:
        next_word = sent[i+1][0]
        next_postag = sent[i+1][1]
        features.update({
            'isupper': next_word[0].isupper(),
            'next_postag': next_postag,
        })
    else:
        features['End'] = True

    return features

# Return Word vector if one exists, else return None
def word2vec_feature(word):
    feature = ('None')
    try:
        feature = (w2v_model.wv[word])
    except KeyError:
        feature = ('None')

    return feature

# Features - Word 2 vec word embeddings
def word2features_w2v(sent, i):
    word = sent[i][0]
    features = {
        'w2v': word2vec_feature(word),
    }

    return features

# Get features for each word in the sentence.
def getSentencefeatures(sentence):
    return [word2features_w2v(sentence, i) for i in range(len(sentence))]

# Get lables for each word in the sentence
def getlabels(sentence):
    return [label for token, postag, label in sentence]

# Get tokens from the sentence
def sent2tokens(sentence):
    return [token for token, postag, label in sentence]

# TP
# B-* => B-*
# I-* => I-*
#
# TN
# O => O
#
# FN
# B-* => O
# I-* => O
#
# FP
# O => B-*
# O => I-*
def compute_results(y_pred, y_test):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(y_pred)):
        for j in range(len(y_test[i])):

            if y_test[i][j].startswith(('B-', 'I-')) and y_pred[i][j].startswith(('B-', 'I-')):
                tp += 1
            if y_test[i][j].startswith('O') and y_pred[i][j].startswith('O'):
                tn += 1
            if y_test[i][j].startswith('O') and y_pred[i][j].startswith(('B-', 'I-')):
                fp += 1
            if y_test[i][j].startswith(('B-', 'I-')) and y_pred[i][j].startswith('O') :
                fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    return precision, recall, accuracy

# Predicts on input test file based on a trained model.
# Specify model_name file
def predict_on_test_set(filepath):
    sentence = []
    test_data = []
    count = 0
    with open(filepath) as chunking_data:
        for line in chunking_data.readlines():

            if 0 == len(line.strip()):
                test_data.append(sentence)
                sentence = []
                continue
            else:
                parts = line.split(' ')
                temp = (parts[0].strip(), parts[1].strip(), parts[2].strip())

            sentence.append(temp)
            temp = ()

    # Get Test sentences with features.
    X_test = [getSentencefeatures(s) for s in test_data]

    # Get Test lables.
    y_test = [getlabels(s) for s in test_data]

    try:
        clf = joblib.load(model_name)
        y_pred = clf.predict(X_test)
        precision, recall, accuracy = compute_results(y_pred, y_test)
        print (model_name.title(), '\nPrecision:', precision , '\nRecall: ', recall, '\nAccuracy:', accuracy)
        labels = list(clf.classes_)
        metrics.flat_f1_score(y_test, y_pred,
                              average='weighted', labels=labels)

        # Sort and Group labels
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )

        print('\nResults from sklearn_crfsuite metrics \n')
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        ))
    except:
        print('There was an exception!',  sys.exc_info()[0])

# Predicts on list of text.
# Returns Chunks from the sentence and and Predicted labels.
def predicts_on_text(text_list):

    # Generate POS Tag for text list
    test_data = nltk.pos_tag(text_list)

    # Get Test sentences with features for sentence.
    # X_test = [getSentencefeatures(s) for s in test_data]
    X_test = [getSentencefeatures(test_data)]

    try:
        clf = joblib.load(model_name)
        y_pred = clf.predict(X_test)

        print ('Chunks: \n')
        chunks = []
        chunk = ''
        tag_index = 0
        for i in range(len(y_pred[0])):
            if tag_index >= len(y_pred[0]):
                break
            if y_pred[0][tag_index].startswith('O'):
                chunks[len(chunks)-1] = chunks[len(chunks)-1] + text_list[tag_index]
                break
            if y_pred[0][tag_index].startswith('B-'):
                chunk += text_list[tag_index] + ' '
                next_tag_index = tag_index + 1
                for j in range(len(y_pred[0])):
                    if next_tag_index >= len(y_pred[0]):
                        chunks.append(chunk)
                        chunk = ''
                        tag_index = next_tag_index
                        break
                    if y_pred[0][next_tag_index].startswith('I-'):
                        chunk += text_list[next_tag_index] + ' '
                        next_tag_index = next_tag_index + 1
                    else:
                        chunks.append(chunk)
                        chunk = ''
                        tag_index = next_tag_index
                        break

        for c in chunks:
            print(c)

        return chunks, y_pred
    except 'AttributeError':
        print('There was an exception!', sys.exc_info()[0])


# Train word2vec model for word embeddings.
def train_word2vec_model_for_word_embeddings(filepath):
    count = get_sentence_count(filepath)
    sentences = [None] * count
    sentence = []
    index = 0
    with open(filepath) as chunking_data:
        for line in chunking_data.readlines():

            if 0 == len(line.strip()):
                sentences[index] = sentence
                index += 1
                sentence = []
                continue
            else:
                parts = line.split(' ')
                temp = parts[0].strip()

            sentence.append(temp)

    w2v_model = Word2Vec(sentences, size=128, window=5, min_count=3, workers=4)

    # Save w2v model.
    joblib.dump(w2v_model, w2v_model_name)

    return w2v_model


# Train a new model based on training file or if there is a change in the feature set
def train_model():
    # Provide input file name
    trainfile = 'chunking_dataset.txt'

    # Provide model name to save
    new_model_name = 'chunking_crf_model_pos_next_previous_word_case_start_word_w2v'

    train_sentences, test_sentences = get_test_train_split_from_file(trainfile, 0.95)

    print(len(train_sentences))
    print(len(test_sentences))

    # Get Training sentence with features.
    X_train = [getSentencefeatures(s) for s in train_sentences]

    # Get Training lables.
    y_train = [getlabels(s) for s in train_sentences]

    # Get Test sentences with features.
    X_test = [getSentencefeatures(s) for s in test_sentences]

    # Get Test lables.
    y_test = [getlabels(s) for s in test_sentences]

    print(getSentencefeatures(train_sentences[0])[0])

    crf = sklearn_crfsuite.CRF(
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )

    crf.fit(X_train, y_train)

    labels = list(crf.classes_)

    # Create a CRF model file
    joblib.dump(crf, new_model_name)

    y_pred = crf.predict(X_test)

    precision, recall, accuracy = compute_results(y_pred, y_test)

    print (precision, recall, accuracy)

    metrics.flat_f1_score(y_test, y_pred,
                          average='weighted', labels=labels)

    # group B and I results
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))

if __name__ == '__main__':

    # w2v_model = train_word2vec_model_for_word_embeddings('chunking_dataset.txt')

    # print(w2v_model.wv['airline'])

    # w2v_model = joblib.load(w2v_model_name)

    # train_model()
    predict_on_test_set('chunking_dataset_1.txt')

    # text_list = ['The','Food','and','Drug','Administration','had','raised','questions','about','the','device', '’s','design', '.']
    # predicts_on_text(text_list)
    #
    # text_list = ['South', 'African', 'economy', 'is', 'a', 'great', 'example', '.']