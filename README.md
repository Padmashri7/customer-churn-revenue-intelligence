# Customer Churn Prediction & Revenue Intelligence

An integrated data science project combining machine learning, customer analytics and revenue intelligence to investigate customer churn, evaluate predictive models and identify financially important customer segments.

## 1. Business Problem

Customer churn affects recurring revenue and long-term business performance. Identifying customers who are likely to leave can help businesses prioritise retention efforts.

However, predicting churn alone does not explain the financial significance of customer attrition.

This project addresses two connected questions:

- **Churn Prediction:** Can machine learning identify customers with a higher likelihood of churning?
- **Revenue Intelligence:** Which customer segments and contract categories account for the largest monthly charges associated with churned customers?

## 2. Project Objectives

- Explore customer characteristics and churn patterns.
- Prepare customer data for machine learning.
- Compare classification models using appropriate evaluation metrics.
- Address class imbalance using class weighting.
- Optimise prediction thresholds using validation data.
- Analyse monthly revenue exposure associated with churn.
- Identify high-value customer segments.
- Translate analytical findings into business recommendations.

## 3. Dataset Overview

The dataset contains 1,000 customer records and 10 columns.

| Attribute | Description |
|---|---|
| CustomerID | Customer identifier |
| Age | Customer age |
| Gender | Customer gender |
| Tenure | Customer tenure |
| MonthlyCharges | Monthly customer charges |
| ContractType | Subscription contract category |
| InternetService | Internet service category |
| TotalCharges | Total customer charges |
| TechSupport | Technical support status |
| Churn | Customer churn status |

**Dataset observations**

- Total records: 1,000
- Total columns: 10
- Duplicate records: 0
- Missing values were identified in `InternetService` and handled during preprocessing.
- Churned customers: 883 (88.3%)
- Active customers: 117 (11.7%)

The dataset has a substantial class imbalance, which makes accuracy alone insufficient for evaluating model performance.

## 4. Exploratory Data Analysis

The analysis examined:

- Customer churn distribution.
- Monthly charges by churn status.
- Tenure and age patterns.
- Contract type and monthly charges.
- Distribution of monthly charges and tenure.
- Relationships between numerical variables.

## 5. Data Preparation & Modelling

The predictive modelling workflow included:

- Encoding the target variable.
- Selecting Age, Gender, Tenure and MonthlyCharges as model features.
- Stratified training, validation and testing splits.
- Standardising numerical features using training data.
- Handling class imbalance through class weighting.
- Comparing prediction probabilities at different classification thresholds.

The final modelling split consisted of 600 training records, 200 validation records and 200 testing records.

## 6. Machine Learning Models

Three classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

Evaluation included:

- Accuracy
- Precision
- Recall
- Macro F1-score
- ROC-AUC
- Average Precision
- Confusion Matrix

Classification thresholds were optimised using validation data before evaluating the models on the test set.

## 7. Model Performance

**Final Test Set Results**

| Model | Accuracy | Precision | Recall | Macro F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 78.50% | 89.41% | 85.88% | 0.5324 | 0.6711 | 0.9451 |
| Random Forest | 76.50% | 96.43% | 76.27% | 0.6427 | 0.8421 | 0.9792 |
| XGBoost | 74.50% | 95.65% | 74.58% | 0.6190 | 0.8206 | 0.9760 |

**Key result**

Random Forest achieved the highest test ROC-AUC (0.8421), precision (96.43%) and macro F1-score (0.6427) among the three models.

Logistic Regression achieved higher churn recall (85.88%), illustrating the trade-off between identifying more churned customers and limiting false-positive predictions.

Random Forest was selected for the saved model, scaler and classification threshold.

## 8. Revenue Exposure Analysis

The second part of the project extends the predictive analysis into customer and revenue intelligence.

The analysis examined monthly charges associated with customers labelled as churned.

| Metric | Result |
|---|---:|
| Total customers | 1,000 |
| Churned customers | 883 |
| Active customers | 117 |
| Total monthly charges | $74,391.29 |
| Charges associated with churned customers | $67,073.36 |
| Churn-associated share of monthly charges | 90.16% |

These figures represent historical monthly charges associated with customers labelled as churned. They should not be interpreted as independently verified future revenue loss.

## 9. Customer Value & Contract Intelligence

### Contract Analysis

Month-to-Month customers accounted for:

- $38,789.60 in churn-associated monthly charges.
- 57.83% of total churn-associated monthly charges.
- An observed churn rate of 100% in this dataset.

### High-Value Customer Analysis

Customers were classified as High Value or Standard Value using the dataset's average monthly charge of $74.39 as the threshold.

Key findings:

- High-value churned customers accounted for $45,097.65 in monthly charges.
- This represented approximately 67.25% of churn-associated monthly charges.
- High-value customers had an observed churn rate of 93.15%, compared with 83.53% for standard-value customers.

### Revenue Concentration

The top 10% of churned customers, comprising 89 customers, accounted for 15.34% of churn-associated monthly charges.

These findings help identify customer groups that may deserve closer retention investigation.

## 10. Business Recommendations

Based on the observed analysis:

1. **Prioritise high-value customers:** Investigate service issues and retention opportunities among customers with higher monthly charges.
2. **Review Month-to-Month contracts:** Examine why this contract group has the highest observed churn and revenue exposure.
3. **Investigate early attrition:** Review the experience of high-paying customers with relatively short tenure.
4. **Use predictive scores carefully:** Apply model predictions to prioritise retention reviews rather than treating them as certain churn outcomes.
5. **Monitor model trade-offs:** Balance missed churn cases against the cost of contacting customers who would otherwise remain.

These are analytical recommendations and hypotheses for further business testing, not proven retention improvements.

## 11. Limitations

- The dataset contains 1,000 customer records and may not represent a broader customer population.
- The churn distribution is highly imbalanced.
- The test set contains only 23 non-churned customers, so performance estimates for that class are particularly sensitive.
- The predictive model uses four selected customer features.
- Revenue exposure is based on historical monthly charges associated with churned customers, not measured future revenue loss.
- Retention campaign effectiveness has not been tested.

## 12. Tools & Technologies

| Category | Tools |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Model Persistence | Joblib |
| Development Environment | Jupyter Notebook |

## 13. Project Files

| File | Description |
|---|---|
| [Customer_Churn.ipynb](Customer_Churn.ipynb) | Integrated customer churn prediction, model evaluation, revenue exposure analysis and customer value intelligence |

## 14. How to Run the Project

1. Clone or download this repository.
2. Install the required Python libraries.
3. Place the `customer_churn_data.csv` dataset in the notebook's working directory.
4. Open `Customer_Churn.ipynb` in Jupyter Notebook or JupyterLab.
5. Run the notebook cells in sequence.

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

**Note:** The dataset is not included in this repository. The notebook expects a local file named `customer_churn_data.csv`.

## 15. Conclusion

This project combines predictive modelling with customer and revenue intelligence.

The machine learning analysis compares three classification algorithms and demonstrates why precision, recall, macro F1-score and ROC-AUC provide more meaningful evaluation than accuracy alone in an imbalanced classification problem.

The business analysis extends the project beyond churn prediction by examining contract-level revenue exposure, customer value segments and revenue concentration.

Together, these analyses provide a framework for identifying customers who may require attention and understanding the financial characteristics of customer attrition.
