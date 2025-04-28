# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load iris Data set

2. Create a DataFrame from the Dataset

3. Add Target Labels to the DataFrame

4. Split Data into Features (X) and Target (y)

5. Split Data into Training and Testing Sets

6. Initialize the SGDClassifier Model

7. Train the Model on Training Data

8. Make Predictions on Test Data

9. Evaluate Accuracy of Predictions

10. Generate and Display Confusion Matrix

11. Generate and Display Classification Report

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by   :  Shyam Sujin U
RegisterNumber :  212223040201
```

```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris

iris=load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['Target']=iris.target

df.head()

X=df.drop(columns='Target')
Y=df['Target']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=1)

model=SGDClassifier(max_iter=1000,tol=0.001)

model.fit(X_train,Y_train)

accuracy=model.predict(X_test)
score=accuracy_score(Y_test,accuracy)
print(f"Accuracy Score is {score}")

conf_mat=confusion_matrix(accuracy,Y_test)
print(conf_mat)

sns.heatmap(df.corr(),annot=True)

```

## Output:

<img width="1625" alt="EXP07" src="https://github.com/user-attachments/assets/22395850-951f-4b4f-9a6e-4416168bcf03" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
