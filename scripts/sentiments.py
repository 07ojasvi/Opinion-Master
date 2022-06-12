import os
import re
import string
import sys
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from textblob import TextBlob
import tweepy
import pycountry
from tabulate import tabulate

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentiText
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer


import nltk
from nltk.corpus import stopwords

from tqdm import tqdm
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

import warnings
warnings.filterwarnings("ignore")

def tokenization(text):
        text = re.split('\W+', text)
        return text

#Removing Punctuation
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

ps = nltk.PorterStemmer()
def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def remove_stopword(x):
    return [y for y in x if y not in stopwords.words('english')]



class Sentiments:

    def __init__(self, searchTerm, numberOfTweets):
        self.keyword = searchTerm
        self.numberOfTweet = numberOfTweets
        nltk.download('stopwords')
        self.reportID = str(uuid.uuid4().hex)

    def generateReport(self):
            

            BASE_PATH=os.path.join("reports/", self.reportID)
            os.mkdir(BASE_PATH)

            consumerKey = 'U63C3XUhvDt8ZfyZCvcIQl7EP'
            consumerSecret = 'CKIAsQhxe3jzFyayBC5QX0k8T7XdaV8MReS4XGcfBGlHnOA5Ni'
            accessToken = '1109478438774071297-pXvNvQftR0MMhlx3jB74hyX937PUaA'
            accessTokenSecret = 'EtF6aBn0moa1Vn2HbUZyIE0Y2zEQ8m1YARgFPrty8cmtR'
            auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
            auth.set_access_token(accessToken, accessTokenSecret)
            #api = tweepy.API(auth) # gives error: Twitter error code 429 with Tweepy
            api = tweepy.API(auth, wait_on_rate_limit=True)

            nltk.download('vader_lexicon')

            def percentage(part,whole):
                return 100 * float(part)/float(whole)
            keyword = self.keyword
            noOfTweet = self.numberOfTweet
            tweets = tweepy.Cursor(api.search_tweets, q=keyword).items(noOfTweet)
            positive = 0
            negative = 0
            neutral = 0
            polarity = 0


            tweet_list = []
            neutral_list = []
            negative_list = []
            positive_list = []

            for tweet in tweets:
                if (not tweet.retweeted) and ('RT @' not in tweet.text):
                    tweet_list.append(tweet.text)
                    analysis = TextBlob(tweet.text)
                    score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
                    print(score)
                    neg = score['neg']
                    neu = score['neu']
                    pos = score['pos']
                    comp = score['compound']
                    polarity += analysis.sentiment.polarity

                    if neg > pos:
                        negative_list.append(tweet.text)
                        negative += 1
                    elif pos > neg:
                        positive_list.append(tweet.text)
                        positive += 1
                    elif pos == neg:
                        neutral_list.append(tweet.text)
                        neutral += 1
                
            positive = percentage(positive, noOfTweet)
            negative = percentage(negative, noOfTweet)
            neutral = percentage(neutral, noOfTweet)
            polarity = percentage(polarity, noOfTweet)
            positive = format(positive, '.1f')
            negative = format(negative, '.1f')
            neutral = format(neutral, '.1f')

            #Number of Tweets (Total, Positive, Negative, Neutral)
            tweet_list = pd.DataFrame(tweet_list)
            neutral_list = pd.DataFrame(neutral_list)
            negative_list = pd.DataFrame(negative_list)
            positive_list = pd.DataFrame(positive_list)
            # print("total number: ",len(tweet_list))
            # print("positive number: ",len(positive_list))
            # print("negative number: ", len(negative_list))
            # print("neutral number: ",len(neutral_list))

            sentiments = []
            tweets_new = []

            #merge, then append

            for index, row in neutral_list.iterrows():
                tweets_new.append(row[0])
                sentiments.append('neutral')

            for index, row in negative_list.iterrows():
                tweets_new.append(row[0])
                sentiments.append('negative')

            for index, row in positive_list.iterrows():
                tweets_new.append(row[0])
                sentiments.append('positive')

            # percentile_list = pd.DataFrame(
            #     {'lst1Title': lst1,
            #      'lst2Title': lst2,
            #      'lst3Title': lst3
            #     })

            train = pd.DataFrame({
                'text': tweets_new,
                'sentiment': sentiments
            })

            train.drop_duplicates()

            tw_list = pd.DataFrame(tweet_list)
            tw_list.head()
            tw_list["text"] = tw_list[0]

            tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
            for index, row in tw_list['text'].iteritems():
                score = SentimentIntensityAnalyzer().polarity_scores(row)
                neg = score['neg']
                neu = score['neu']
                pos = score['pos']
                comp = score['compound']
                if neg > pos:
                    tw_list.loc[index, 'sentiment'] = "negative"
                elif pos > neg:
                    tw_list.loc[index, 'sentiment'] = "positive"
                else:
                    tw_list.loc[index, 'sentiment'] = "neutral"
                tw_list.loc[index, 'neg'] = neg
                tw_list.loc[index, 'neu'] = neu
                tw_list.loc[index, 'pos'] = pos
                tw_list.loc[index, 'compound'] = comp

            df = pd.DataFrame(tweet_list)

            csvName = '0_csv.csv'
            csvPath = os.path.join(BASE_PATH, csvName)
            df.to_csv(csvPath, index=False)

            #Creating new data frames for all sentiments (positive, negative and neutral)

            tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]
            tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]
            tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]

            #Function for count_values_in single columns

            def count_values_in_column(data,feature):
                total=data.loc[:,feature].value_counts(dropna=False)
                percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
                return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

            

            html1 = count_values_in_column(tw_list,"sentiment").to_html()
        
            # write html to file
            countPath = os.path.join(BASE_PATH, "1_count.html")
            text_file = open(countPath, "w")
            text_file.write(html1)
            text_file.close()

            # create data for Pie Chart
            pc = count_values_in_column(tw_list,"sentiment")
            names= pc.index
            size=pc["Percentage"]
            
            # Create a circle for the center of the plot
            my_circle=plt.Circle( (0,0), 0.7, color='white')
            plt.pie(size, labels=names, colors=['#66CDAA','#FFF68F','#FFB6C1'])
            p=plt.gcf()
            savePath = os.path.join(BASE_PATH, "2_pie.png")

            p.gca().add_artist(my_circle)

            p.savefig(savePath, format='png')
            #my_circle.figure.savefig(savePath, format='png')

            #Function to Create Wordcloud

            def create_wordcloud(text):
                mask = np.array(Image.open("static/cloud.png"))
                stopwords = set(STOPWORDS)
                wc = WordCloud(background_color="white",
                            mask = mask,
                            max_words=3000,
                            stopwords=stopwords,
                            repeat=True)
                return wc.generate(str(text))
                # wc.to_file("wc.png")
                # print("Word Cloud Saved Successfully")
                # path="wc.png"

            #Creating wordcloud for positive sentiment
            positiveWCPath = os.path.join(BASE_PATH, "3_positiveWC.png")
            create_wordcloud(tw_list_positive["text"].values).to_file(positiveWCPath)

            #Creating wordcloud for negative sentiment
            negativeWCPath = os.path.join(BASE_PATH, "4_negativeWC.png")
            create_wordcloud(tw_list_negative["text"].values).to_file(negativeWCPath)

            #Creating wordcloud for neutral sentiment
            neutralWCPath = os.path.join(BASE_PATH, "5_neutralWC.png")
            create_wordcloud(tw_list_neutral["text"].values)

            #Calculating tweet's lenght and word count
            tw_list['text_len'] = tw_list['text'].astype(str).apply(len)
            tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))

            length = pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()).to_html()
            sentimentLengthPath = os.path.join(BASE_PATH, "6_sentimentLength.html")
            text_file = open(sentimentLengthPath, "w")
            text_file.write(length)
            text_file.close()

            sentimentWordCount = pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()).to_html()
            sentimentWordCountPath = os.path.join(BASE_PATH, "7_sentimentWordCount.html")
            text_file = open(sentimentWordCountPath, "w")
            text_file.write(sentimentWordCount)
            text_file.close()

            

            tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))

            #Appliyng tokenization
            
            tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))

            #Removing stopwords
            
                
            tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))

            #Appliyng Stemmer
            tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))

            #Cleaning Text
            def clean_text(text):
                text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
                text_rc = re.sub('[0-9]+', '', text_lc)
                tokens = re.split('\W+', text_rc)    # tokenization
                text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
                return text

            list8 = tw_list.head().to_html()
            list8Path = os.path.join(BASE_PATH, "8_list.html")
            text_file = open(list8Path, "w")
            text_file.write(list8)
            text_file.close()

            #Appliyng Countvectorizer
            countVectorizer = CountVectorizer(analyzer=clean_text) 
            countVector = countVectorizer.fit_transform(tw_list['text'])
            print('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))
            #print(countVectorizer.get_feature_names())


            count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())

            # Most Used Words
            count = pd.DataFrame(count_vect_df.sum())
            countdf = count.sort_values(0,ascending=False).head(20)
            mostUsed = countdf[1:11].to_html()

            mostUsedPath = os.path.join(BASE_PATH, "9_mostUsed.html")
            text_file = open(mostUsedPath, "w")
            text_file.write(mostUsed)
            text_file.close()

            #Function to ngram
            def get_top_n_gram(corpus,ngram_range,n=None):
                vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
                bag_of_words = vec.transform(corpus)
                sum_words = bag_of_words.sum(axis=0) 
                words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
                return words_freq[:n]

            #n2_bigram
            n2_bigrams = get_top_n_gram(tw_list['text'],(2,2),20)
            n2BigramTable = tabulate(n2_bigrams, tablefmt='html')

            bigramPath = os.path.join(BASE_PATH, "10_bigram.html")
            text_file = open(bigramPath, "w")
            text_file.write(n2BigramTable)
            text_file.close()

            #n3_trigram
            n3_trigrams = get_top_n_gram(tw_list['text'],(3,3),20)

            n3TrigramTable = tabulate(n3_trigrams, tablefmt='html')

            triramPath = os.path.join(BASE_PATH, "11_trigram.html")
            text_file = open(triramPath, "w")
            text_file.write(n3TrigramTable)
            text_file.close()

            #Below is a helper Function which generates random colors which can be used to give different colors to your plots.F
            def random_colours(number_of_colors):
                '''
                Simple function for random colours generation.
                Input:
                    number_of_colors - integer value indicating the number of colours which are going to be generated.
                Output:
                    Color in the following format: ['#E86DA4'] .
                '''
                colors = []
                for i in range(number_of_colors):
                    colors.append("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
                return colors

            train.dropna(inplace=True)

            tw_list = pd.DataFrame(tweet_list)
            tw_list["text"] = tw_list[0]
            tw_list #:)

            temp = train.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
            table12 = temp.style.background_gradient(cmap='Purples').render()

            table12Path = os.path.join(BASE_PATH, "12_table.html")
            text_file = open(table12Path, "w")
            text_file.write(table12)
            text_file.close()

            plt.figure(figsize=(8,6))
            countplot = sns.countplot(x='sentiment',data=train)
            fig = countplot.get_figure()
            countplotPath = os.path.join(BASE_PATH, "13_countplot.png")
            fig.savefig(countplotPath) 


            def plotlySaveImage(plotlyFigure, path):
                plotlyFigure.write_image(path)

            def dataFrameSave(figure, path):
                text_file = open(path, "w")
                text_file.write(figure)
                text_file.close()

            fig = go.Figure(go.Funnelarea(
                text =temp.sentiment,
                values = temp.text,
                title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
            ))

            funnelPath = os.path.join(BASE_PATH, "14_funnel.png")
            plotlySaveImage(fig, funnelPath)

            train['text'] = train['text'].apply(lambda x:clean_text(x))

            train['temp_list'] = train['text'].apply(lambda x:str(x).split()) #changed selected text to text
            top = Counter([item for sublist in train['temp_list'] for item in sublist])
            temp = pd.DataFrame(top.most_common(20))
            temp.columns = ['Common_words','count']
            commonWords = temp.style.background_gradient(cmap='Blues').to_html()
            commonWordsPath = os.path.join(BASE_PATH, "15_commonWords.html")
            dataFrameSave(commonWords, commonWordsPath)

            fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 
                    width=700, height=700,color='Common_words')
            commonWordsPath = os.path.join(BASE_PATH, "16_commonWords.png")
            plotlySaveImage(fig, commonWordsPath)

            
            train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))

            top = Counter([item for sublist in train['temp_list'] for item in sublist])
            temp = pd.DataFrame(top.most_common(20))
            temp = temp.iloc[1:,:]
            temp.columns = ['Common_words','count']
            commonWords = temp.style.background_gradient(cmap='Purples').to_html()
            commonWordsPath = os.path.join(BASE_PATH, "16_commonWords.html")
            dataFrameSave(commonWords, commonWordsPath)

            fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')
            treePath = os.path.join(BASE_PATH, "17_tree.png")
            plotlySaveImage(fig, treePath)

            train['temp_list1'] = train['text'].apply(lambda x:str(x).split()) #List of words in every row for text
            train['temp_list1'] = train['temp_list1'].apply(lambda x:remove_stopword(x)) #Removing Stopwords

            Positive_sent = train[train['sentiment']=='positive']
            Negative_sent = train[train['sentiment']=='negative']
            Neutral_sent = train[train['sentiment']=='neutral']


            #MosT common positive words
            top = Counter([item for sublist in Positive_sent['temp_list'] for item in sublist])
            temp_positive = pd.DataFrame(top.most_common(20))
            temp_positive.columns = ['Common_words','count']
            mostPositive = temp_positive.style.background_gradient(cmap='Greens').to_html()

            mostPositivePath = os.path.join(BASE_PATH, "18_mostPositive.html")
            dataFrameSave(mostPositive, mostPositivePath)

            return self.reportID
                
        
        





        























    
















    
