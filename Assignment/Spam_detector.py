import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import auc, roc_curve, confusion_matrix, classification_report, precision_recall_curve




def Output(Imputation, spam_value, bnbcm , bnbReport, AccuracyScore, roc_auc, Mnbcm, MnbReport, MnbAccuracyScore, mnbroc_auc):

    print()
    print("Start Report:")
    print()
    print("Imputation Technique to check Null values: ")
    print(Imputation)
    print()
    print("Spam is ",spam_value, "%")
    print()
    print("Bernoulli Confusion Matrix: ", bnbcm)
    print()
    print(" Bernoulli Classification Report: ")
    print(bnbReport)
    print()
    print("Accuracy of Bernoulli: ", AccuracyScore,"%" )
    print()
    print("Bernoulli ROC Accuracy: ",roc_auc, "%")
    print()
    print("Confusion Matrix: ",Mnbcm)
    print()
    print(" Multinomial Classification Report: ")
    print(MnbReport)
    print()
    print("Accuracy of Multinomial: ", MnbAccuracyScore,"%" )
    print()
    print("Multinomial ROC Accuracy: ",mnbroc_auc, "%")
    print()


def PlotGraph(false_positive_rate, true_positive_rate, precision, recall, mnbfalse_positive_rate, mnbtrue_positive_rate, mnbprecision, mnbrecall ):
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Bernoulli ROC')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.savefig('Bernoulli_roc_curve.png', dpi=200)


    plt.clf()
    # create plot
    plt.plot(precision, recall, label='Bernoulli Precision-recall curve')
    _ = plt.xlabel('Precision')
    _ = plt.ylabel('Recall')
    _ = plt.title('Bernoulli Precision-recall curve')
    _ = plt.xlim([0.735, 1])
    _ = plt.ylim([0, 1.02])
    _ = plt.legend(loc="lower left")

    # save figure
    plt.savefig('Bernoulli_precision_recall.png', dpi=200)
    plt.clf()


    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Multinomial ROC')
    plt.plot(mnbfalse_positive_rate, mnbtrue_positive_rate)
    plt.savefig('Multinomial_roc_curve.png', dpi=200)
    plt.clf()

   
    # create plot
    plt.plot(mnbprecision, mnbrecall, label='Multinomial Precision-recall curve')
    _ = plt.xlabel('Precision')
    _ = plt.ylabel('Recall')
    _ = plt.title('Multinomial Precision-recall curve')
    _ = plt.xlim([0.735, 1])
    _ = plt.ylim([0, 1.02])
    _ = plt.legend(loc="lower left")

    # save figure
    plt.savefig('Multinomial_precision_recall.png', dpi=200)
    plt.clf()




def main():
    data =  pd.read_csv("spam_ham_dataset.csv")
    data['label_num'] = data.label.map({'ham': 1 , 'spam' : 0}) 

    # Create Training and Test data set
    X=data.text
    y=data.label_num

    #Check Null values
    Imputation = data.isnull().sum()
   
    spam_ham_value = data.label.value_counts()

    #Spam Value before spliiting
    spam_value = (spam_ham_value[1]/float(spam_ham_value[0]+spam_ham_value[1]))*100
   
    #Splitting between Train and Test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)
    
    vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_train_vectorized.toarray().shape

    modelBernoulli = BernoulliNB(alpha=0.1)
    modelBernoulli.fit(X_train_vectorized, y_train)

    # Bernoulli Predictions, Probability and Accuracy
    bnbpredictions = modelBernoulli.predict(vectorizer.transform(X_test))
    bnbprobability = modelBernoulli.predict_proba(vectorizer.transform(X_test))
    AccuracyScore = metrics.accuracy_score(y_test, bnbpredictions) * 100

    # Bernoulli Confusion Matrix, Classification Report and ROC
    bnbcm = confusion_matrix(y_test, bnbpredictions)
    bnbReport = classification_report(y_test, bnbpredictions)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, bnbprobability[:,1])
    roc_auc = auc(false_positive_rate, true_positive_rate) * 100
    precision, recall, thresholds = precision_recall_curve(y_test, bnbprobability[:,1])

    
    modelMultiNomial = MultinomialNB(alpha=0.1)
    modelMultiNomial.fit(X_train_vectorized, y_train)

    # Multinomial Predictions, Probability and Accuracy
    Mnbpredictions = modelMultiNomial.predict(vectorizer.transform(X_test))
    Mnbprobability = modelMultiNomial.predict_proba(vectorizer.transform(X_test))
    MnbAccuracyScore = metrics.accuracy_score(y_test, Mnbpredictions) * 100
    
    # Multinomial Confusion Matrix, Classification Report and ROC
    Mnbcm = confusion_matrix(y_test, Mnbpredictions)
    MnbReport = classification_report(y_test, Mnbpredictions)
    mnbfalse_positive_rate, mnbtrue_positive_rate, thresholds = roc_curve(y_test, Mnbprobability[:,1])
    mnbroc_auc = auc(mnbfalse_positive_rate, mnbtrue_positive_rate) * 100
    mnbprecision, mnbrecall, thresholds = precision_recall_curve(y_test, Mnbprobability[:,1])


    #Print output
    Output(Imputation, spam_value, bnbcm , bnbReport, AccuracyScore, roc_auc, Mnbcm, MnbReport, MnbAccuracyScore, mnbroc_auc )

    # Print the graphs
    PlotGraph(false_positive_rate, true_positive_rate, precision, recall, mnbfalse_positive_rate, mnbtrue_positive_rate, mnbprecision, mnbrecall )

      
   
    



if __name__ == "__main__":
    main()    