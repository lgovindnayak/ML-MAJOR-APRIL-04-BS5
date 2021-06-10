from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests   
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
from nltk.tokenize.toktok import ToktokTokenizer
from pyngrok import ngrok
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vs = SentimentIntensityAnalyzer()

url = "https://inshorts.com/en/read/sports"
news_data = []
news_category = url.split('/')[-1]
data = requests.get(url)
soup = BeautifulSoup(data.content)

urls = ['https://inshorts.com/en/read/sports',
        'https://inshorts.com/en/read/world',
        'https://inshorts.com/en/read/politics']

def build_dataset(urls):
  news_data = []
  for url in urls:
    news_category = url.split('/')[-1]
    data = requests.get(url)
    soup = BeautifulSoup(data.content)
    # here some concept of html 
    news_articles = [{'news_headline':headline.find('span',attrs={"itemprop":"headline"}).string,
                      'news_article':article.find('div',attrs={'itemprop':'articleBody'}).string,
                      'news_category':news_category}
                     
                     for headline,article in zip(soup.find_all('div',class_=["news-card-title news-right-box"]),
                                                 soup.find_all('div',class_=["news-card-content news-right-box"]))]
    news_articles = news_articles[0:20]
    news_data.extend(news_articles)

  df = pd.DataFrame(news_data)   
  df = df[["news_headline","news_article","news_category"]]
  return df   

df = build_dataset(urls)
import nltk 
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


# 1. Lower case
df.news_headline = df.news_headline.apply(lambda x:x.lower())
df.news_article = df.news_article.apply(lambda x:x.lower())

# 2. HTMP Tags
df.news_headline = df.news_headline.apply('html_tag')
df.news_article = df.news_article.apply('html_tag')

# 3. Contractions
df.news_headline = df.news_headline.apply(con)
df.news_article = df.news_article.apply(con)

# 4. Special Charcters
df.news_headline = df.news_headline.apply('remove_sp')
df.news_article = df.news_article.apply('remove_sp')

# 5. Stop Words
df.news_headline = df.news_headline.apply(remove_stopwords)
df.news_article = df.news_article.apply(remove_stopwords)

df['compound'] = df['news_article'].apply(lambda x: vs.polarity_scores(x)['compound'])

data_for_pred = 'exaustralia captain ricky ponting said australia got find player like ms dhoni hardik pandya kieron pollard finishing role best batsmen bat top four big bash explained ponting suggested names glenn maxwell mitchell marsh marcus stoinis position'
vs.polarity_scores(data_for_pred)


analyzer = SentimentIntensityAnalyzer()


st.title("Sentimental Analysis Using Lexicon Based Approach...")

iput = st.text_input("Enter Text:")
oput_dict =  analyzer.polarity_scores(iput)

if st.button('Analyze'):
  if oput_dict['compound'] >= 0.05:
    st.write(' *This is Positive review* :smile:')
  elif oput_dict['compound'] <= -0.05:
    st.write('*This is Negative review* :angry:')
  else:
    st.write('*This is Neutral review* :unamused:')
