# Customer_Churn_Prediction
# Customer Churn Prediction

Losing a customer is expensive, and losing them by surprise is worse. This project explores how far a fairly simple neural network can get in predicting which telecom customers are about to churn — using nothing but the account and service details a company already has on file: contract type, tenure, monthly charges, and the services they've signed up for.

It's built on the classic [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset and walks through the full pipeline: cleaning the raw data, encoding it for a model, training a small ANN, and honestly evaluating how well (and where) it actually works.

## Approach

- Exploratory analysis — looked at how tenure and monthly charges differ between churned and retained customers using histograms.
- Cleaning — dropped the customer ID column, fixed 11 rows where TotalCharges was blank (new customers with zero tenure), and standardized inconsistent category labels ("No internet service" to "No", etc.).
- Encoding — converted binary Yes/No columns to 1/0, label-encoded gender, and one-hot encoded the multi-category columns (InternetService, Contract, PaymentMethod).
- Scaling — min-max scaled tenure, MonthlyCharges, and TotalCharges so they're on the same footing as the binary features.
- Model — a small feed-forward neural network (Keras/TensorFlow): one hidden layer of 20 ReLU units, a sigmoid output for binary classification, trained for 100 epochs with the Adam optimizer.
- Evaluation — classification report and confusion matrix on a held-out 20% test split.

## Results

<img width="1180" height="419" alt="image" src="https://github.com/user-attachments/assets/58ad8de5-e48e-43ab-b388-8b567ea84d3f" />


Overall test accuracy: about 79%

The model is noticeably better at spotting customers who'll stay than those who'll leave — expected, since churners are the minority class here (about 27% of the dataset). Recall on the churn class (0.59) is the main thing worth improving next.

## Tech stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras

## Possible next steps

- Handle class imbalance (e.g. class weights or SMOTE) to lift churn-class recall
- Try tree-based models (Random Forest, XGBoost) as a baseline comparison
- Feature importance or SHAP to explain why a customer is flagged
- Hyperparameter tuning and cross-validation instead of a single train/test split

## Dataset

IBM Telco Customer Churn dataset — 7,043 customers, 20 features, binary churn label. Available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
