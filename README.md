# Employee Attrition Prediction using Machine Learning
## Project Overview
This is a self-initiated machine learning project focused on analyzing HR data to identify key factors influencing employee attrition and predicting whether an employee is likely to leave the organization.

The project aims to support data-driven workforce management and employee retention strategies.

## Dataset
- IBM HR Analytics Employee Attrition Dataset
- 1,470 employee records
- 35+ features including demographics, job role, satisfaction, compensation, and work-life factors
- Target variable: Attrition (Yes / No)

## Exploratory Data Analysis (EDA)
Key insights identified through EDA include:
- Approximately 16% employee attrition rate
- Higher attrition among employees working overtime
- Lower job satisfaction strongly correlates with attrition
- Younger employees (25–35 age group) show higher turnover
- Sales and HR departments exhibit higher attrition rates

## Machine Learning Models Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Model Performance
| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 82%      | 0.79      | 0.75   | 0.77     |
| Decision Tree       | 80%      | 0.77      | 0.73   | 0.75     |
| Random Forest       | 85%      | 0.83      | 0.81   | 0.82     |

The Random Forest model delivered the best performance.

## Key Features Influencing Attrition
- OverTime
- Job Satisfaction
- Monthly Income
- Years at Company

## Tools & Technologies
- Python (Pandas, NumPy, Scikit-learn)
- Matplotlib, Seaborn
- Jupyter Notebook
- Power BI (for dashboarding)

## Business Insights & Recommendations
- Reduce overtime to improve work-life balance
- Improve job satisfaction through engagement initiatives
- Monitor high-risk groups such as younger employees and sales staff
- Use predictive insights to proactively manage attrition

## Future Enhancements
- Implement explainable AI (SHAP / LIME)
- Predict time-to-attrition using survival analysis
- Deploy the model as a web-based HR analytics application
