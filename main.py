from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfTransformer
app = Flask(__name__)
import pickle


# open a file, where you stored the pickled data
file = open('nb_clf_crude_oil', 'rb')
clf = pickle.load(file)
file.close()

file = open('vectorizer_crude_oil', 'rb')
vec = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        print(request.form)
        myDict = request.form
        text_anal = (myDict['text'])
        text_anal=[text_anal]
        topic_anal = int(myDict['topic'])
        # print(text_anal)
        # print(topic_anal)
        
        # inputFeatures = [fever, pain, age, runnyNose, diffBreath]
        
        X_vec_test=vec.transform(text_anal)  #not use fit_transform bcoz the model is already trained. if we use again treats the new words as new and calculations become wrong
        X_vec_test=X_vec_test.todense()
        
        
        tfidf=TfidfTransformer()
        X_tfidf_test=tfidf.fit_transform(X_vec_test)
        X_tfidf_test=X_tfidf_test.todense()

        #print(X_tfidf_test)
        text_sentiment =clf.predict(X_tfidf_test)
        
        # print(text_sentiment)
        sentiment = ' '.join([str(elem) for elem in text_sentiment])
        return render_template('show.html', inf=sentiment)
    return render_template('index.html')
    

if __name__ == "__main__":
    app.run(debug=True)