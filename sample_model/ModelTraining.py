import pandas as pd
import numpy as np
import re
import unihandecode
import string
from string import digits
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import math
import random
from nltk import word_tokenize
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score 
from sklearn.metrics import precision_recall_fscore_support
import logging
import pickle

## reading subsequent email data
fileName = '200 Tickets Scored data.xlsx'

#returns a dataframe
def read_train_test(fileName) :
    subs = pd.read_excel(fileName)
    return subs


def text_clean(x):
    x = re.sub("<.*?>","", x)                                          #remove html tags
    x = re.sub(r'([^\s\w]|_)+','',x)                                   #retain only alphanumeric chars and space
    x = unihandecode.unidecode(x)                                      #Convert latin to ASCII characters
    x = x.lower()                                                      #convert to lower case characters
    x = x.translate(str.maketrans("","",string.punctuation))           #removing punctuation marks
    x = x.translate(str.maketrans("","",digits))                       #removing numbers
    x = x.strip()                                                      #removing leading and trailing white spaces
    x = PorterStemmer().stem(x)                                        #stem to root word
    #x = x.replace(" ", "")                                            #remove white space
    return x   



def stopWordRemoval(data1, pathToStopwords = 'file_stopword_new.txt') :
    f = open(pathToStopwords, 'r')
    f.seek(0)
    stopwords_new = [(i.split('\n'))[0] for i in f.readlines()]
    # remove stopwords
    data2 = data1

    res=[]
    for i in data2:
        iq = i.split()
        resultwords  = [word for word in iq if word not in stopwords_new]
        result = ' '.join(resultwords)
        res.append(result)
    
    data2 = res
    return data2

def splitTrainTest(dataSize) :
    ## 75% of the sample size
    smp_size = math.floor(0.75 * dataSize)
    random.seed(123)
    train_ind = random.sample(range(0,dataSize),smp_size)
    test_ind=[]
    for x in range(0,dataSize):
        if x not in train_ind:
            test_ind.append(x)

    return train_ind, test_ind

def splitter(data):
    # data = data[['Number', 'Date_Of_Interact', 'Notes_flown_to_Customer']]
    
    data_sorted = data.sort_values(['Number', 'Date_Of_Interact'])
    number_Set, numbers = list(set(list(data_sorted['Number']))), list(data_sorted['Number'])
    
    count_list = []
    k=1
    for i in sorted(number_Set):
        for j in numbers:
            if i == j:
                count_list.append(k)
                k += 1
        k=1

    data_sorted['unique_Counts'] = count_list
    data_sorted_initials, data_sorted_subsequent = data_sorted[data_sorted['unique_Counts'] <= 2], data_sorted[data_sorted['unique_Counts'] > 2]
    
    data_sorted_initials = data_sorted_initials[['Number', 'Date_Of_Interact', 'Notes_flown_to_Customer']]
    data_sorted_subsequent = data_sorted_subsequent[['Number', 'Date_Of_Interact', 'Notes_flown_to_Customer']]

    return data_sorted_initials, data_sorted_subsequent



def extract_etr(data):
    import nltk
    nltk.download('wordnet')

    timeWords = ['minute', 'minutes', 'min', 'mins', 'hour', 'hr','hours', 'h', 'hrs', 'days', 'day' ]

    if len(str(data)) < 1:
        print("error:"+str(data))
        return ''
 
    #tokenize the data so that we get base form of the words, 
    #and then reverse the list. Reversing helps in getting the number of hours or minutes easily.

    words=nltk.word_tokenize(data)
    words.reverse()
 
    lemmatizer=nltk.WordNetLemmatizer()
    processed_data=[]

    print("Processing:"+data)

    for i,org_word in enumerate(words):
        print(org_word)
        word=lemmatizer.lemmatize(org_word)

        #check if the word matches any of the possible minute or hour versions, 
        #if it matches, then return the "original" form of the word, i.e non lemmatized form.
        if org_word.lower() in timeWords :
            if words[i+1].isdigit() :
                return str(words[i+1]) + ' ' + org_word
            else :
                return ''


def train(trainFeature) :
    # text clean
    subs = read_train_test(fileName)
    #subs, subs_subsequent = splitter(subs)
    queryStr = trainFeature + ' == 0 | ' + trainFeature + ' == 1 '
    subs = subs.query(queryStr)
    data1 = subs['Notes_flown_to_Customer'].astype(str)
    data1 = data1.apply(lambda x:text_clean(x))
    data2 = stopWordRemoval(data1)

    # create corpus
    x1 = [word_tokenize(x) for x in data2]

    # document term matrix
    cv = CountVectorizer()
    d = cv.fit_transform(data2)
    dtm = pd.DataFrame(d.toarray(),columns=cv.get_feature_names())
    # dtm_m = dtm.as_matrix()

    train_ind, test_ind = splitTrainTest(dataSize=len(data2))

    df = subs[[trainFeature,'Notes_flown_to_Customer']]

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    #for x in train_ind:
    df_train = df.iloc[train_ind]


    #for y in test_ind:
    df_test = df.iloc[test_ind]

    dtm_train = pd.DataFrame()
    dtm_test = pd.DataFrame()

    #for x in train_ind:
    dtm_train = dtm.iloc[train_ind]
    
    #for y in test_ind:
    dtm_test = dtm.iloc[test_ind]

    #TODO review logic
    #df_train[trainFeature] = df_train[trainFeature].map(lambda x:0 if np.isnan(x) else x)
    #df_test[trainFeature] = df_test[trainFeature].map(lambda x:0 if np.isnan(x) else x)
    #df[trainFeature] = df[trainFeature].map(lambda x:0 if np.isnan(x) else x)

    model = GaussianNB()
    model.fit(dtm_train,df_train[trainFeature])
    pred = model.predict(dtm)
    print('Feature ' + str(trainFeature) + ' feature size ' + str(len(subs)))
    print(model.score(dtm,df[trainFeature]))
    if len(list(set(df[trainFeature]))) == 2:
        confmat = confusion_matrix(pred, df[trainFeature])
        confmat = pd.DataFrame(confmat, index=['Predicted 0','Predicted 1'],columns=['Actual 0','Actual 1'])
        precision, recall, fscore, support = precision_recall_fscore_support(df[trainFeature], pred, average='binary')

        print(confmat)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))


    return cv, model




features = ['i_link_stat_chk', 'i_logs', 'i_cust_dep', 'i_poa', 'i_svc_rstr', 'i_etr']

def trainAndStoreModels(outputFile, vocabOutputFile) :
    import nltk
    nltk.download('punkt')

    outputModels = {}
    outputCVs = {}
    cv = None
    for feature in features :
        cv, model = train(feature)
        outputModels[feature] = model
        outputCVs[feature] = cv

    fp = open(outputFile, 'wb')
    pickle.dump(outputModels, fp)

    fpCV = open(vocabOutputFile, 'wb')
    pickle.dump(outputCVs, fpCV)


def predictModelStr(emailModelPklFile, vocabModelPklFile, inputStr) :
    fpCV = open(vocabModelPklFile, 'rb')
    featureVocabs = pickle.load(fpCV)

    fpModel = open(emailModelPklFile,'rb')
    modelDict = pickle.load(fpModel)

    for feature in features :
        featureNames = featureVocabs[feature].get_feature_names()
        newCV = CountVectorizer(vocabulary=featureNames)
        x = newCV.fit_transform([inputStr])
        dtm_Predict = pd.DataFrame(x.toarray(), columns = newCV.get_feature_names())
    
        model = modelDict[feature]
        output = model.predict(dtm_Predict)
        print('feature = ' + str(feature) + ' output = ' + str(output))
        if feature == 'i_etr' :
            i_etr_time = extract_etr(inputStr)
            print('feature = i_etr_time ' + ' output = ' + str(i_etr_time))


def predictModel(emailModelPklFile, vocabModelPklFile, data, path_to_utils) :

    fpCV = open(vocabModelPklFile, 'rb')
    featureVocabs = pickle.load(fpCV)

    fpModel = open(emailModelPklFile,'rb')
    modelDict = pickle.load(fpModel)

    data = data[['Number', 'Date_Of_Interact', 'Notes_flown_to_Customer']]

    # Removing all the entries where string length is less than 10. 
    data['Notes_flown_to_Customer'] = data['Notes_flown_to_Customer'].astype('str')
    indexes = (data['Notes_flown_to_Customer'].str.len() > 10)
    data = data.loc[indexes]

    # Splitting the initials and subsequent mails.
    data, data_subsequent = splitter(data)

    data1 = data['Notes_flown_to_Customer'].astype(str)
    data1 = data1.apply(lambda x: text_clean(x))
    path_to_utils = path_to_utils + 'file_stopword_new.txt'
    data2 = stopWordRemoval(data1, pathToStopwords=path_to_utils)

    for feature in features :
        featureNames = featureVocabs[feature].get_feature_names()
        newCV = CountVectorizer(vocabulary=featureNames)
        dtm = newCV.fit_transform(data2)
        dtm_Predict = pd.DataFrame(dtm.toarray(), columns = newCV.get_feature_names())
    
        model = modelDict[feature]
        data[feature] = model.predict(dtm_Predict)

        if feature == 'i_etr' :
            i_etr_time = extract_etr(inputStr)
            data['i_etr_time'] = i_etr_time
    
    return data


def main() :
    emailModelPklFile = 'emailmodels.pkl'
    vocabModelPklFile = 'vocab.pkl'

    #trainAndStoreModels(emailModelPklFile, vocabModelPklFile)
    predictModelStr(emailModelPklFile, vocabModelPklFile, 'Dear customer ,Link is affected due to media down  in TCL network.ETR 3 hours.')


def test() :
    global features 
    features = ['i_logs', 'i_cust_dep']
    emailModelPklFile = 'emailmodels_test.pkl'
    vocabModelPklFile = 'vocab_test.pkl'
    trainAndStoreModels(emailModelPklFile, vocabModelPklFile)
    predictModelStr(emailModelPklFile, vocabModelPklFile, 'Dear customer ,Link is affected due to media down  in TCL network.ETR 3 hours.')
    predictModelStr(emailModelPklFile, vocabModelPklFile, '''BGLR_MYTINTEL8FL3_01 




VCG-1-8-109
Start Time	End Time	Idle Seconds	ES	SES	UAS	Valid Frames Transmitted	Valid Frames Received	Valid Bytes Transmitted	Valid Bytes Received	Core Header Single Error Corrections	Type Header Single Error Corrections	Core Header CRC Errors	Type Header CRC Errors	Payload FCS Errors	VCG Interval Valid
06-Jun-2019 11:00:01	06-Jun-2019 11:15:01	0	0	0	0	1019857	867418	541272527	652487096	0	0	0	0	0	1
06-Jun-2019 10:45:02	06-Jun-2019 11:00:02	0	0	0	0	789446	1040526	291870250	911592892	0	0	0	0	0	1
06-Jun-2019 10:30:01	06-Jun-2019 10:45:01	0	0	0	0	773566	587332	349651217	276727080	0	0	0	0	0	1
06-Jun-2019 10:15:02	06-Jun-2019 10:30:02	0	0	0	0	796962	688290	383790370	435752164	0	0	0	0	0	1
06-Jun-2019 10:00:01	06-Jun-2019 10:15:01	0	1	0	0	744388	760382	270012845	541995900	84	0	23155	2	2	1
06-Jun-2019 09:45:02	06-Jun-2019 10:00:02	0	0	0	0	721684	1067343	261150255	924603146	0	0	0	0	0	1
06-Jun-2019 09:30:02	06-Jun-2019 09:45:02	0	0	0	0	805576	622539	469831239	359369406''')


if __name__ == "__main__":
    #test()
    main()
