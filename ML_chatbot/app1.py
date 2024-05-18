from flask import Flask, render_template, request ,session
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

app.secret_key = '1234'

training = pd.read_csv('Training.csv')
testing= pd.read_csv('Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']
testy    = le.transform(testy)

clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
# print(clf.score(x_train,y_train))hi

# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
# print ( scores)
print ("Accuracy for Decision Tree classifier: ",scores.mean())


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}


for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index


def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        description_list={}
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('Symptom_severity.csv') as csv_file:
        severityDictionary={}
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _diction={row[0]:int(row[1])}
            severityDictionary.update(_diction)



def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        precautionDictionary={}
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4],row[5],row[6]]}
            precautionDictionary.update(_prec)

def getInfo():
    print("Hi, I'am MedBot")
    print("\nYour Name? \t\t\t\t", end="->")
    name = input("")
    print("Hello, ", name)

def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if(len(pred_list) > 0):
        return 1, pred_list
    else:
        return 0, []

def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])



def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))


def tree_to_code(tree, feature_names, symptom, num_days):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    questions = []
    options = []

    while True:
        conf,cnf_dis=check_pattern(chk_dis,symptom)
        if conf==1:
            questions.append("searches related to input:")
            options.append(cnf_dis)
            break
        else:
            questions.append("Enter valid symptom.")
            options.append([])

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == symptom:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            for syms in list(symptoms_given):
                question = f"Are you experiencing {syms}? : "
                questions.append(question)
                options.append(["yes", "no"])

    recurse(0, 1)

    return questions, options



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    symptom = request.form['symptom']
    days = int(request.form['days'])
    questions, options = tree_to_code(clf, cols, symptom, days)


    # Save days value in the session
    session['days'] = days


    # Preprocess data for rendering in the template
    question_option_pairs = list(zip(questions, options))

    
    return render_template('index.html', name=name, question_option_pairs=question_option_pairs)

# @app.route('/result', methods=['POST'])
# def result():
#     # Get the submitted form data
#     form_data = request.form
#     symptoms_exp=[]

#     # Retrieve days value from the session
#     days = session.get('days')

#     # Loop through the form data to find the question with "yes" selected
#     for question, option in form_data.items():
#         if option.lower() == "yes":
#             # Print the question to the terminal
#             words = question.split()
#             q2 = words[-2].strip('?')
#             symptoms_exp.append(q2)
#     print(symptoms_exp)
#     second_prediction=sec_predict(symptoms_exp)
#     print(second_prediction)

 

#     return "Question printed to terminal"


@app.route('/result', methods=['POST'])
def result():
    # Get the submitted form data
    form_data = request.form
    symptoms_exp=[]

    # Retrieve days value from the session
    days = session.get('days')

    # Loop through the form data to find the question with "yes" selected
    for question, option in form_data.items():
        if option.lower() == "yes":
            # Print the question to the terminal
            words = question.split()
            q2 = words[-2].strip('?')
            symptoms_exp.append(q2)
    print(symptoms_exp)
    second_prediction=sec_predict(symptoms_exp)
    print(second_prediction)
    
    # Calculate condition
    calc_condition(symptoms_exp, days)

    return "Second Prediction printed to terminal"


getSeverityDict()
getDescription()
getprecautionDict()

if __name__ == '__main__':
    app.run(debug=True)





