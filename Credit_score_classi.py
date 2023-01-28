import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

data = pd.read_csv("train.csv")
print(data.head())


print(data.info()) #For having a look at the information about the columns in the dataset:

"""Now we will check for null values in our data"""

print(data.isnull().sum()) #No values are null in our dataset """

"""The dataset doesnt have any null values.
As this dataset is labelled, lets have a look at the Credit_Score column values:"""

print(data["Credit_Score"].value_counts())

"""Data Exploration

The dataset has many features that can train a Machine Learning model for credit score classification. 
Lets explore all the features one by one.

I will start by exploring the occupation feature to know if the occupation 
of the person affects credit scores:"""

ml = data.columns.values.tolist()

for i in range(6,len(ml)-1):
    
    if (type(data.iloc[2,i]) != np.float64) and ml[i] != ('Type_of_Loan' or 'Payment_of_Min_Amount'):

        fig = px.box(data,x=ml[i],color="Credit_Score", title=f"Credit Scores Based on {ml[i]}]",
                     color_discrete_map={'Poor':'red', 'Standard':'yellow','Good':'green'})
        fig.show()


    elif (type(data.iloc[2,i]) == np.float64):

        A = data[ml[i]].loc[data.Credit_Score == "Good"]
        B = data[ml[i]].loc[data.Credit_Score == "Standard"]
        C = data[ml[i]].loc[data.Credit_Score == "Poor"]


        box = plt.boxplot([A,B,C], patch_artist=True,labels = ["Good","Standard","Poor"])
        plt.title(ml[i],loc='center')
        plt.ylabel(ml[i])
        
        colors = ['lightgreen', 'yellow', 'red']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.legend([box["boxes"][0], box["boxes"][1],box["boxes"][2]], ["Good","Standard","Poor"], loc='upper left')
        plt.show()

'''Since Credit Mix is a categorical variable, so need to change it to numerical variable to use in ML model'''
data["Credit_Mix"] = data["Credit_Mix"].map({"Standard": 1, "Good": 2, "Bad": 0})

from sklearn.model_selection import train_test_split
x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Credit_Mix", "Outstanding_Debt", 
                   "Credit_History_Age", "Monthly_Balance"]])
y = np.array(data[["Credit_Score"]])

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.33, random_state=42)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

print("Credit Score Prediction : ")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) : ")
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
print("Predicted Credit Score = ", model.predict(features))


