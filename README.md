# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, drop unnecessary columns, and encode categorical variables. 
2. Define the features (X) and target variable (y). 
3. Split the data into training and testing sets. 
4. Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: chandru k
RegisterNumber:  212224220017
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
<img width="1227" height="370" alt="image" src="https://github.com/user-attachments/assets/3a8f7fe5-0cea-4c51-8872-1922255a29c9" />
<img width="1239" height="399" alt="image" src="https://github.com/user-attachments/assets/422b5177-cd9f-49c4-81ba-ac4491987b4f" />
<img width="982" height="664" alt="image" src="https://github.com/user-attachments/assets/b9cc59ac-53cf-41bb-966a-12e36785c5b8" />
<img width="954" height="286" alt="image" src="https://github.com/user-attachments/assets/be99e449-4770-4eb5-a570-38672e0a8f6c" />
<img width="974" height="178" alt="image" src="https://github.com/user-attachments/assets/dfd51d38-b6ad-433a-89c2-ba173604340a" />
<img width="978" height="212" alt="image" src="https://github.com/user-attachments/assets/b74bc25d-2df4-4527-9be6-3217d432e96f" />
<img width="975" height="471" alt="image" src="https://github.com/user-attachments/assets/9e3ee49d-1f38-4335-afed-57d8e61c54b7" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
