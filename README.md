# Bank Loan Predictor

## The Dataset

I have chosen the dataset named `bank_loan_approval` from Kaggle. The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. It includes various factors such as:

- **CIBIL Score**
- **Income**
- **Employment Status**
- **Loan Term**
- **Loan Amount**
- **Assets Value**
- **Loan Status**

## Aim of Project

The primary aim of this project is to develop a robust and accurate predictive model for bank loan approval using machine learning algorithms, specifically:

- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Naive Bayes**

The project seeks to leverage these algorithms to classify loan applications based on their likelihood of default, ensuring that the bank can make informed, data-driven decisions that minimize financial risk and maximize customer satisfaction.

## Description of the Algorithms

- **K-Nearest Neighbors (KNN)**: KNN is a simple yet effective supervised learning algorithm used for classification and regression tasks. It assigns a class label or predicts a value based on the majority vote or average of its nearest neighbors in the feature space.

- **Naive Bayes**: Naive Bayes is a probabilistic classifier based on Bayes' theorem with an assumption of independence between features. It calculates the probability of each class given a set of features and selects the class with the highest probability as the prediction.

- **Decision Tree**: Decision Tree is a versatile supervised learning algorithm capable of performing classification and regression tasks. It partitions the feature space into a tree structure and makes decisions by recursively splitting the data based on the features that provide the most information gain or decrease in impurity.

## Pair Plot and Correlation Matrix

- **Pair Plot**: These plots help identify relationships between various features and their relationship with the output labels or class labels column in the dataset.
- **Correlation Matrix**: A correlation matrix is a table that shows the correlation coefficients between a set of variables. Each variable is compared to the others, and the resulting correlation coefficient is a value between -1 and 1. A correlation matrix is often used to investigate the relationship between different variables in a dataset.

## Data Preprocessing

There are some categorical features/attributes which are in object datatype; we need to convert those to numeric data for better analysis. For that, we will use encoding. There are various methods for encoding categorical data, including one-hot encoding, label encoding, and target encoding. We will use label encoding here.

| Variable (Feature) | Description           |
|--------------------|-----------------------|
| **Education**      | Graduate (0), Not Graduate (1) |
| **Self Employed**  | Yes (1), No (0)       |
| **Loan Status**    | Approved (0), Rejected (1) |

- In our dataset, there are no null values present in any of the features, and there are no duplicate values present in the dataset.

## Feature Selection

The feature `loan_id` was removed (dropped) as it is not useful in our analysis and will not be used by any machine learning algorithm applied.

## Evaluation Metrics (Accuracy, Precision, F1 Score, Confusion Matrix)

Accuracy, precision, and F1 score are commonly used metrics in data mining to evaluate the performance of classification models.

1. **Accuracy** measures the proportion of correctly classified instances among all instances. Its formula is:

    ```
    Accuracy = Number of Correct Predictions / Total Number of Predictions
    ```

2. **Precision** measures the proportion of correctly predicted positive cases among all predicted positive cases. Its formula is:

    ```
    Precision = True Positives / (True Positives + False Positives)
    ```

3. **F1 Score** is the harmonic mean of precision and recall, providing a balance between the two metrics. It is calculated as:

    ```
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    ```

4. **Confusion Matrix** is a table used to evaluate the performance of a classification model. It presents a summary of the model's predictions against the actual values in a tabular format. The matrix typically includes counts of true positive, true negative, false positive, and false negative predictions, providing insights into the model's accuracy, precision, recall, and other performance metrics.

## Applying KNN, Naive Bayes, and Decision Tree Machine Learning Algorithms

- To apply these algorithms, we first separate the class label from our dataset and then divide our dataset and class labels into two parts: the training dataset (on which our model will be trained) and the testing dataset (which will be used to test our model performance).
- For the Decision Tree, the maximum depth of the tree formed from train data is taken as 8.
- For K-Nearest Neighbors, the value of k (nearest neighbors) is taken as 5.

## Conclusion

### Accuracy Scores (Test Data):
- **K-Nearest Neighbors**: 90.07%
- **Naive Bayes**: 93.07%
- **Decision Tree**: 97.57%

### Precision Scores (Test Data):
- **K-Nearest Neighbors**: 86.65%
- **Naive Bayes**: 89.02%
- **Decision Tree**: 98.48%

### F1 Scores (Test Data):
- **K-Nearest Neighbors**: 87.07%
- **Naive Bayes**: 91.15%
- **Decision Tree**: 96.76%

I have applied three Machine Learning algorithms on our dataset (K-Nearest Neighbors, Naive Bayes, and Decision Tree). Out of all these, the best performing algorithm for our dataset in terms of accuracy, precision, and F1 Scores is Decision Tree. The worst-performing algorithm in terms of accuracy, precision, and F1 Scores is K-Nearest Neighbors.
