#!/usr/bin/python3

import numpy as np
import pandas as pd
import yaml as y
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

def get_long_input():
  lines = []
  print("Please input your data and press <enter> twice to continue")
  
  while True:
      line = input()
      if line:
          lines.append(line)
      else:
          break
  
  long_input = '\n'.join(lines)
  
  if(len(lines) == 8):
    return long_input
  else:
    print("Sorry, you must input the data correctly")
    print('''
Please input like this:

'Pregnancies' : 6
'Glucose' : 148
'BloodPressure' : 72
'SkinThickness' : 35
'Insulin' : 0
'BMI' : 33.6
'DiabetesPedigreeFunction' : 0.627
'Age' : 50''')
    exit(0)
  

def intro():
  string = '''
Please wait a moment!

This program is a machine learning-based tool that aims to detect the likelihood of an individual having diabetes. 
By utilizing a dataset containing relevant information such as pregnancies, glucose, blood_press, skin_thic, insulin, bmi, dpf, age. 
The program employs advanced classification algorithms to make accurate predictions.
  '''
  print(string)

def main():
  intro()
  # loading the diabetes dataset to a pandas DataFrame
  diabetes_dataset = pd.read_csv('diabetes.csv')

  diabetes_dataset['Outcome'].value_counts()
  diabetes_dataset.groupby('Outcome').mean()

  # separating the data and labels
  X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
  Y = diabetes_dataset['Outcome']

  """Train Test Split"""
  X_train, X_test, Y_train, Y_test = train_test_split(X.values,Y, test_size = 0.2, stratify=Y, random_state=2)

  """Training the Model"""
  classifier = svm.SVC(kernel='linear')

  #training the support vector Machine Classifier
  classifier.fit(X_train, Y_train)

  # accuracy score on the training data
  X_train_prediction = classifier.predict(X_train)
  training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

  # accuracy score on the test data
  X_test_prediction = classifier.predict(X_test)
  test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


  """Making a Predictive System"""
  input_data = get_long_input()

  try:
    input_data = y.load(input_data, Loader=y.FullLoader)

    pregnancies = input_data['Pregnancies']
    glucose = input_data['Glucose']
    blood_press = input_data['BloodPressure']
    skin_thic = input_data['SkinThickness']
    insulin = input_data['Insulin']
    bmi = input_data['BMI']
    dpf = input_data['DiabetesPedigreeFunction']
    age = input_data['Age']

    data = (pregnancies, glucose, blood_press, skin_thic, insulin, bmi, dpf, age)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = classifier.predict(input_data_reshaped)
  
    if (prediction[0] == 0):
      print('The person is not diabetic')
    else:
      print('The person is diabetic')
  
  except:
    print("\nSorry, an error has occured.")


if __name__ == "__main__":
  main()
