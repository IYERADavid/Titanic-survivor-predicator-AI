import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("titanic.csv")
df.insert(7,'male', df['Sex'] == 'male')

x = df[["Pclass","male","Age","Siblings/Spouses","Parents/Children","Fare"]].values
y = df["Survived"].values

model = LogisticRegression()
model.fit(x, y)
percent = round(model.score(x, y)*100, 1)
print("______Titanic survivor predicator._______ \n")
print("our prediction result is expected to be true at " + str(percent) + " % \n")
print("Note: you must answer 1 instaed of Yes and 0 instead of No \n")

pcl = int(input("Pclass : "))
sex = int(input("IT is a boy : "))
age = int(input("Age : "))
sbl_spo = int(input("Number of Siblings/Spouses : "))
par_chil = int(input("Number of parents/children : "))
fare = int(input("Fare in dollars : "))
person_data = [[pcl, sex, age, sbl_spo, par_chil, fare]]
result = model.predict([person_data][0])
gender = "He" if sex == 1 else "She" 
if result == 1:
        print(" => " + gender + " survived.")
else:
    print(" => " + gender + " didn't survive")
