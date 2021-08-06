import requests
from bs4 import BeautifulSoup
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
 
# A list for all these things
url_list=[]
date_time=[]
news_text=[]
headlines=[]

# extracting urls of all the news 
# available on the first 3 pages and saving to url_list
i=6
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
data_pred = pd.DataFrame({ 'Headline':headlines,
                       'News': news_text,
                     })


######################################################################################
#Testing

nb_clf=pickle.load(open("nb_clf_crude_oil","rb"))
vectorizer=pickle.load(open("vectorizer_crude_oil","rb"))

#predicting using trained classifier
X_test=data_pred.iloc[:,1]          #column with news articles
X_vec_test=vectorizer.transform(X_test)  #not use fit_transform bcoz the model is already trained. if we use again treats the new words as new and calculations become wrong
X_vec_test=X_vec_test.todense()

#transform data by applying term frequency inverse document freq(TF-IDF)
tfidf=TfidfTransformer()
X_tfidf_test=tfidf.fit_transform(X_vec_test)
X_tfidf_test=X_tfidf_test.todense()


#predict the sentiment values
y_pred= nb_clf.predict(X_tfidf_test)




