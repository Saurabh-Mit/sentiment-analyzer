# Using VADAR for sentiment analysis of crude oil price using "https://oilprice.com/"

import requests
from bs4 import BeautifulSoup
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# A list for all these things
url_list=[]
date_time=[]
news_text=[]
headlines=[]

# extracting urls of all the news 
# available on the first 3 pages and saving to url_list
for i in range(1,3):
    url='https://oilprice.com/Energy/Crude-Oil/Page-{}.html'.format(i)
    request=requests.get(url)
    soup=BeautifulSoup(request.text,"html.parser")
    for links in soup.find_all('div',{'class':'categoryArticle'}):
        for info in links.find_all('a'):
            if info.get('href') not in url_list:
                url_list.append(info.get('href'))
 

# processing extracted links to get info from them
for www in url_list:
    #process each url
    #getting headlines
    headlines.append(www.split("/")[-1].replace("-"," "))
    request=requests.get(www)
    soup=BeautifulSoup(request.text,"html.parser")
    
    
    #store date and time of publication
    for dates in soup.find_all('span',{'class':'article_byline'}):
        date_time.append(dates.text.split('-')[-1])
        
    #store text of news
    temp=[]
    for news in soup.find_all('p'):
        temp.append(news.text)
        
    #identify last line of article to identify end of news to remove unnecessary info
    for last_sentence in reversed(temp):
        if last_sentence.split(" ")[0]=="By" and last_sentence.split(" ")[-1]=="Oilprice.com":
            break
        elif last_sentence.split(" ")[0]=="By":
            break
        
    #pruning non informative news and coming data b/w last_sentence and starting sentence
    joined_text=' '.join(temp[temp.index("More Info")+1:temp.index(last_sentence)])
    news_text.append(joined_text)
    

#all the required info is now extracted .
#creating all this into a dataframe
news_df=pd.DataFrame({ 'Date': date_time,
                       'Headline':headlines,
                       'News': news_text,
                     })


#using VADER for data analysis
analyser=SentimentIntensityAnalyzer() 

def compound_score(text):
    return analyser.polarity_scores(text)["compound"]

news_df["sentiment"]=news_df["News"].apply(compound_score)
                