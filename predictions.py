import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

df = pd.read_csv("titanic.csv")
df.insert(7,'male', df['Sex'] == 'male')

main_x = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
x = df[['Pclass', 'male', 'Age']].values
y = df["Survived"].values

# what the kfold bellow help us is to find the range of our predicton 
# level, so most of the time our percent will be between 77 t0 80

scores = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print(scores)

model = LogisticRegression()
model.fit(main_x, y)
percent = round((np.mean(scores))*100, 1)
print("______Titanic survivor predicator._______ \n")
print("our prediction result is expected to be true at a level of " + str(percent) + " % \n")
print("Note: you must answer 1 instaed of Yes and 0 instead of No \n")

pcl = int(input("Pclass : "))
sex = int(input("IT is a boy : "))
age = int(input("Age : "))
sbl_spo = int(input("Number of Siblings/Spouses : "))
par_chil = int(input("Number of parents/children : "))
fare = float(input("Fare in dollars : "))
person_data = [[pcl, sex, age, sbl_spo, par_chil, fare]]
result = model.predict(person_data)[0]
gender = "He" if sex == 1 else "She" 
if result == 1:
        print(" => " + gender + " survived.")
else:
    print(" => " + gender + " didn't survive")
