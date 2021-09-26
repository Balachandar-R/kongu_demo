###TextPharse Classification (Given Text into Class1(Software) & Class2(Hardware))
###Two Class classification using BernaulliNB()
import csv 
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def load_file():
    with open('D:\\Python\\NLP\\test_csv.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        reader.next()
        data =[]
        target = []
        for row in reader:
            if row[0] and row[1]:
                data.append(row[0])
                target.append(row[1])
        return data,target

def preprocess():
    """preprocess creates the term frequency matrix for the review data set"""
    data,target = load_file()
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)   
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data) 
    return tfidf_data,count_vectorizer

def learn_model(data,target,data_for_test_tfidf,inputdesc):
    """ preparing data for split validation. 60% training, 40% test"""
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.4,random_state=40)
    ##Building a classifer using NavieBayes Algorithm
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict(data_test)
    
    joblib.dump(classifier, 'Classifier.pkl') 
    
    """For the Demostration purpose the sample Ticket Description must be entered by the user and test it on this classifier """
    demo_predicted1 = classifier.predict(data_for_test_tfidf)  
    print "*********************************"
    for desc, category in zip(inputdesc, demo_predicted1):
       print('%r => %s' % (desc, demo_predicted1))
       
    print "*********************************"  
    #evaluate_model(target_test,predicted)
    return target_test,predicted

def evaluate_model(target_true,target_predicted):
    """Evaluvate the Model """
    print "Classification of Request description(Using Bernoulli NaievBayes Algo)"
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))

   
def main():
    """Driver Fucntion to call other functions """
    data,target = load_file()
    tf_idf,count_vectorizer = preprocess()   

    """Feature Vector created as a seperate pickle file """
    features = count_vectorizer.get_feature_names()
    joblib.dump(features,'Vocabulary.pkl')

    answer=True
    while(answer):
         print("""
               1.Enter your Description to Classify
               2.Display confusion Matrix(Classifier)
               3.Exit/Quit
               """)
         answer=raw_input("What would you like to do?")
         if answer=="1":
             inputdesc= [raw_input("Enter the Request Description to classify:")] 
             data_for_test = count_vectorizer.transform(inputdesc)
             data_for_test_tfidf = TfidfTransformer(use_idf=False).transform(data_for_test)
             target_test,predicted=learn_model(tf_idf,target,data_for_test_tfidf,inputdesc)
         elif answer=="2":
            data,target = load_file()
            tf_idf,count_vectorizer = preprocess() 
            evaluate_model(target_test,predicted)
         elif answer=="3":
             print("\n Goodbye") 
             answer = None
         else:
            print("\n Not Valid Choice Try again")
    
main()
