import numpy as np
import pandas as pd
import csv
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import os
import datetime


class sentiClassifier(object):
    
    def __init__(self,training_file,clf_type):

        self.data = training_file
        #converts csv into pandas DataFrame
        self.processed_data = self.process()
        self.available_clfs = ["mnb"]
        self.clf_type = clf_type

        if clf_type not in self.available_clfs:
            raise ValueError("%s is not a recognized classifier type" % clf_type)
        else:
            self.clf = self.train()    

    def __str__(self):
        return "clf: {0}\ntraining_file: {1}\nlast_training: {2}".format(self.clf_type,self.data,self.date)

    def process(self): 

        df = pd.DataFrame(columns=["Sentiment","Text"])
        index = 0

        with open(self.data,"r") as ts:
            content = csv.reader(ts)
            
            for row in content:
                    
                    if len(row) > 1:
                        remove_whitespace = []
                        for item in row:
                           remove_whitespace.append(item.strip()) 

                        formattedStr = " ".join(remove_whitespace)
                        removed_quotation = re.sub(r'"',"",formattedStr)
                        sentiment = re.match(r'^\d',removed_quotation).group(0)
                        final = re.sub(r'\d\t',"",removed_quotation)
                        
                    else:
                        removed_quotation = re.sub(r'"',"",row[0])
                        sentiment = re.match(r'^\d',removed_quotation).group(0)
                        final = re.sub(r'\d\t',"",removed_quotation)

              
                
                    df.loc[index] = [sentiment,final]

                    index+=1

        return df

    def tfidf(self,data,transform=False):

            if not transform:
                tf_vec = TfidfVectorizer(stop_words="english")
                tf_vec.fit_transform(data)
                return tf_vec
            else:
                return self.fitted_vectorizer.transform(data)


    def train(self):

        X = self.processed_data["Text"]
        y = self.processed_data["Sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

        if self.clf_type == "mnb":

            self.fitted_vectorizer = self.tfidf(X_train)
            self.term_matrix = self.fitted_vectorizer.transform(X_train)
            mnb=MultinomialNB()
          
            mnb.fit(self.term_matrix,y_train)
            self.model = mnb
            self.xtest = self.tfidf(X_test,transform=True)
            self.actual = y_test
            self.date = str(datetime.datetime.now().day) + "/" + str(datetime.datetime.now().month) + "/" + str(datetime.datetime.now().year)

    def accuracy(self):
        pred = self.model.predict(self.xtest)
        prediction_values = np.array(pred)
        actual_values = np.array(self.actual)
        if len(actual_values) == len(prediction_values):
            accurate_ = 0
            for i in range(len(prediction_values)):
                if prediction_values[i] == actual_values[i]:
                    accurate_+=1
            return float(accurate_)/len(prediction_values)
        else:
            raise AssertionError("Prediction values do not match the length of Actual values")

    def dump(self,*args):
        '''
            Arguments must be filenames with the following extensions: .pkl
            First argument must be filename for the fitted tfidf vectorizer
            Second argument must be the filename for the fitted model
        '''
        if len(args) == 2:
            joblib.dump(self.fitted_vectorizer,args[0])
            joblib.dump(self.model,args[1])

            return "Successfully dumped!"


if __name__ == "main":
    classifier = sentiClassifier("training_set.csv","mnb")
    accuracy = classifier.accuracy()
    print(accuracy)
    dump = classifier.dump("fitted_vec.pkl","mnb_mod.pkl")
    print(dump)
