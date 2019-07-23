import pandas as pd
import numpy as np
def sentiment_preprocessor(f1, f2, threshold = 1):
    raw = pd.read_csv(f1, sep = '\t')
    # print(raw.columns)

    #remove duplicated items
    raw = raw[raw.duplicated(subset = 'id', keep = False) == False]
    # print(len(raw))
    # print(raw.head())

    # raw2 = pd.read_excel("./user_tweet.xlsx")
    # print(raw2.columns)
    # print(raw2['tweet_id'].value_counts())
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(raw2[raw2['tweet_id']==870435330549088000])

    raw3 = pd.read_excel(f2)
    raw3 = raw3[raw3.duplicated(subset = 'id', keep = False) == False]


    total_text = pd.merge(raw, raw3, on='id')[['text', 'sentiment']]
    total_text['sentiment'] = total_text['sentiment'] + 1

    testfilt = np.random.randn(len(total_text)) > threshold
    testset = total_text[testfilt]
    trainset = total_text[testfilt == False]

    valfilt = np.random.randn(len(trainset)) > threshold
    valset = trainset[valfilt]
    trainset = trainset[valfilt == False]

    return(
        list(total_text['text']),
        list(zip(testset['text'], testset['sentiment'])),
        list(zip(valset['text'], valset['sentiment'])),
        list(zip(trainset['text'], trainset['sentiment']))
    )