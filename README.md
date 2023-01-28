# Credit-Score-Classifier

Credit Score Classification

Overview
This project aims to develop a machine learning model that can accurately classify an individual's credit score between ["Good", "Standard","Poor"]. The model was trained using a supervised learning algorithm, Random Forest, on a dataset of credit score data.

Problem Statement
The problem is to predict credit score of an individual based on various features such as income, credit history, number of credit inquiries etc. This can be useful for financial institutions to better assess the creditworthiness of potential borrowers.

Results
The model achieved an accuracy of 85% on the test set. Additionally, I was able to achieve a precision of 87% and recall of 90% for the high-risk credit score category. One interesting insight was that the number of credit inquiries in the past six months had a strong correlation with an individual's credit score.

How to Use
To use the model, you can run the Credit_score_classi.py file. You need to provide inputs about your financials such as Annual Income, EMI, Credit history etc which will help our model predict or we should say classify your Credit Score between ["Good", "Standard","Poor"]

Dependencies
~Python 3
~Numpy
~Pandas
~Scikit-learn
~Matplotlib
