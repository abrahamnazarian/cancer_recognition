#!/usr/bin/python


from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from flask import Flask, render_template
import numpy as np
from sklearn.model_selection import learning_curve
import cPickle
import pandas as pd
from sklearn.model_selection import cross_val_score,KFold

#start logging errors
logf = open("loggings.log", "w")

#initialize Flask
app = Flask(__name__)

#Run ML
def supervise_learning(classifier,parameters,df):
    try:
        X=df.iloc[:, 0:-1]
        Y=df.iloc[:, -1]
        if classifier=="SVC":
            parameters['probability']=True;
            clf = SVC(**parameters)
        if classifier == "KNeighborsClassifier":
            clf = KNeighborsClassifier(**parameters)
        if classifier == "RandomForestClassifier":
            clf = RandomForestClassifier(**parameters)
        if classifier == "DecisionTreeClassifier":
            clf = DecisionTreeClassifier(**parameters)
        if classifier == "LogisticRegression":
            clf = LogisticRegression(**parameters)
        if classifier == "GaussianNB":
            clf = GaussianNB(**parameters)
        clf.fit(X, Y);
    except Exception as e:
        logf.write(str(e))
    # save the classifier by fitting on all data
    with open('my_dumped_classifier.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)
    #calculate metrics based on cross validation - 5 fold
    accuracy=cross_val_score(clf, X, Y, cv=5, scoring='accuracy');     accuracy = float("{0:.2f}".format(np.mean(accuracy)))
    precision = cross_val_score(clf, X, Y, cv=5, scoring='precision'); precision = float("{0:.2f}".format(np.mean(precision)))
    recall = cross_val_score(clf, X, Y, cv=5, scoring='recall'); recall = float("{0:.2f}".format(np.mean(recall)))
    f1 = cross_val_score(clf, X, Y, cv=5, scoring='recall'); f1 = float("{0:.2f}".format(np.mean(f1)))
    print('Accuracy: ' + str((accuracy)));
    print('Precision: ' + str((precision)));
    print('Recall: ' + str((recall)));
    print('F1 Score: ' + str((f1)));
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    train_sizes, train_scores, valid_scores = learning_curve(clf, X_train, y_train,
                                                             train_sizes=[int(0.2 * len(X_train)),
                                                                          int(0.4 * len(X_train)),
                                                                          int(0.6 * len(X_train)),
                                                                          int(0.8 * len(X_train))], cv=5);
    roc_data=np.array([]);
    # Compute ROC curve and ROC area for each class
    if classifier!="RandomForestClassifier":
        y_score = clf.predict_proba(X_test)
        fpr,tpr,_= metrics.roc_curve(y_test,y_score[:,1])
        roc_data=np.zeros((len(fpr),2));
        roc_data[:,0]=fpr;
        roc_data[:,1]=tpr;
        print ('roc_data',roc_data.tolist())

    if classifier == "RandomForestClassifier":
        features_list=df.columns;
        importances = clf.feature_importances_
        print ('importances',importances)
        indices = np.argsort(importances)[::-1];
        X=X_train;
        data=[];
        for f in range(X.shape[1]):
            row=[features_list[(indices[f])], importances[indices[f]]];
            data.append(row);

        @app.route("/")
        def index():
            return render_template("feature_importance.html",data=data,train_scores=(train_scores[:,0]).tolist(),valid_scores=(valid_scores[:,0]).tolist(),roc_data=roc_data.tolist())
    else:
        @app.route("/")
        def index():
            return render_template("learning_curve.html",train_scores=(train_scores[:,0]).tolist(),valid_scores=(valid_scores[:,0]).tolist(),roc_data=roc_data.tolist())
    return (1);


#extract data from CSV
df = pd.read_csv('Training_data.csv',na_values='0')
df=df.fillna(0)

clf="KNeighborsClassifier"; parameters={"n_neighbors":5}
clf="DecisionTreeClassifier"; parameters={"max_depth":5}
clf="SVC"; parameters={"C":1.5,"kernel":"rbf"}
#clf="LogisticRegression"; parameters={"penalty":"l2","C":1.0}
clf="GaussianNB"; parameters={};
#clf="RandomForestClassifier"; parameters={"n_estimators":50}

supervise_learning(clf,parameters,df);

def prediction(CSV_file_path):
    # load the trained classifier
    with open('my_dumped_classifier.pkl', 'rb') as fid:
        clf = cPickle.load(fid);
    #extract data from CSV
    data = pd.read_csv('sample_data_for_prediction.csv', na_values='0')
    data = data.fillna(0)
    #do prediction
    y_predicted = clf.predict(data[0:]);
    #prepare the results in a HTML table
    table="<table id='data'><tr>";
    for k in data.columns.values:
        table += "<th>" + str(k) + "</th>";
    table += "<th>Prediction</th>";
    table += "</tr>";
    rowIndex=-1;
    for index,row in data.iterrows():
        table=table+"<tr>";
        for k in data.columns.values:
            table=table+"<td>"+str(row[k])+"</td>";
        table=table+"<td align='center' style='background-color:Orange'>"+str(int(y_predicted[index]))+"</td>";
        table = table + "</tr>";
    @app.route("/predict")
    def predict():
        return render_template("prediction.html", prediction_results=table)


#do the prediction
prediction("sample_data_for_prediction.csv");


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5004)