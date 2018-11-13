import nltk
from sklearn_crfsuite import metrics

chunkGram = r"""
 NP: {<NN><NN.>+}     
 NP: {<JJ>+<NN>+<NN.?>}
 NP: {<NN.?><CC><NN.?>}
 NP: {<NNP><POS><NN>*}
 NP: {<JJ><JJ>}
 NP: {<DT>?<JJ>*<NN>}
 VP: {<VBD>}
 VP: {'<VB.>?<ADV>*<VB.?>+'}
 """

# Get all the unique chunk tags
def get_chunk_tags(y_test):
    chunk_tags = []

    for y in y_test:
        for tag in y:
            if tag not in chunk_tags:
                chunk_tags.append(tag)

    return chunk_tags

def predict_on_test_set(filepath):
    data_list = []
    temp = ()
    predicted_sentences = []
    test_sentences = []
    test_sentence = []
    with open(filepath) as chunking_data:
        for line in chunking_data.readlines():
            parts = line.split(' ')
            if 0 == len(line.strip()):
                chunked_sentence = predict_on_texts(data_list)
                # chunked_sentence.append('')
                predicted_sentences.append(chunked_sentence)
                # test_sentence.append('')
                test_sentences.append(test_sentence)
                # test_sentences.append('')
                # for c in chunked_sentence:
                #     print (c)
                # print('\n')
                data_list = []
                test_sentence = []
                continue
            temp = (parts[0], parts[1])
            # temp1 = (parts[0].strip(), parts[2].strip())

            if len(line.strip()) == 0:
                continue

            data_list.append(temp)
            # test_sentence.append((parts[0].strip(), parts[2].strip()))
            test_sentence.append(parts[2].strip())
            temp = []

    precision, recall, accuracy = compute_results(predicted_sentences, test_sentences)
    print('Precision:', precision, 'Recall: ', recall, 'Accuracy:', accuracy)
    return predicted_sentences

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

    chunk_tags = get_chunk_tags(y_test)
    chunk_tags = sorted(
        chunk_tags,
        key=lambda name: (name[1:], name[0])
    )

    print(metrics.flat_classification_report(
        y_test, y_pred, labels=chunk_tags, digits=3
    ))

def predict_on_texts(tagged):
    # print(tagged)
    chunked_sentence = []
    flag = False
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)

    for chunk in chunked:
        # print(len(chunk))
        try:
            if  'NP' in chunk._label:
                for i in range(len(chunk)):
                    if flag == False:
                        # chunked_sentence.append((chunk[i][0], chunk[i][1], 'B-NP'))
                        print(chunk[i][0], chunk[i][1], 'B-NP')
                        chunked_sentence.append('B-NP')
                        flag = True
                    else:
                        # chunked_sentence.append((chunk[i][0], chunk[i][1], 'I-NP'))
                        print(chunk[i][0], chunk[i][1], 'I-NP')
                        chunked_sentence.append('I-NP')
            else:
                if 'VP' in chunk._label:
                    for i in range(len(chunk)):
                        if flag == False:
                            # chunked_sentence.append((chunk[i][0], chunk[i][1], 'B-VP'))
                            print(chunk[i][0], chunk[i][1], 'B-VP')
                            chunked_sentence.append('B-VP')
                            flag = True
                        else:
                            # chunked_sentence.append((chunk[i][0], chunk[i][1], 'I-VP'))
                            print(chunk[i][0], chunk[i][1], 'I-VP')
                            chunked_sentence.append('I-VP')
                # part = chunk.split('/')
                # chunked_sentence.append((part))
                print('I-' + chunk[0][1])
                chunked_sentence.append('I-' + chunk[0][1])
                flag = False
        except AttributeError:
            print(chunk[1])
            chunked_sentence.append(chunk[1])
            flag = False
            continue

    return chunked_sentence

if __name__ == '__main__':
    chunked_sentences = predict_on_test_set('chunking_dataset_1.txt')
    print ('chunked_sentences : ', chunked_sentences)