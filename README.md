# Predict Employee Attrition Rate

## Description

This project aims to build predictive models to forecast the employee attrition rate within an organization. The goal is to identify employees with a high likelihood of leaving the company, allowing management to implement retention strategies effectively.

Our project is just a developing model, we will keep improve it. Hope you have a great overview with it.
## Data Source

The data used for this project consists of employee records, including information such as age, position, salary, and other factors that may influence the decision to leave. The dataset can be found in the `Raw DataSet` folder.
Data source from : https://www.hackerearth.com/challenges/new/competitive/hackerearth-machine-learning-challenge-predict-employee-attrition-rate/
This is a great competitive, maybe you wanna try !?!

## Folder Structure

```plaintext
Predict-employee-attrition-rate/
│
├── .idea/                       # IDE configuration files
│
├── Processed Data/              # Processed data ready for analysis
│
├── Raw DataSet/                 # Raw datasets
│   └── employee_data.csv        # CSV file containing employee data
│
├── Evaluation.ipynb             # Jupyter Notebook for model evaluation
│
├── Finding_Para_Nemo.ipynb      # Jupyter Notebook for hyperparameter tuning
│
├── Preprocess.ipynb             # Jupyter Notebook for data preprocessing
│
├── Training.ipynb               # Jupyter Notebook for model training
│
├── best_xgb_model.pkl           # Pickle file of the best XGBoost model
│
├── final_model.pkl              # Final model after training
│
└── trained_xgb_model.pkl        # Trained XGBoost model
```

## Model and Training

We had build own XGBoost model. During the initial phase of developing the XGBoost model for predicting employee attrition, an initial set of hyperparameters was utilized. However, the performance of the model was not satisfactory, indicating that the chosen parameters did not yield the desired results.

## Challenges Identified
- **Underfitting**: The model showed signs of underfitting, as evidenced by a high bias and poor performance on both training and validation datasets.
- **Improper Hyperparameters**: The initial hyperparameters (e.g., learning rate, max depth, and regularization) were not optimally configured, leading to subpar predictive accuracy.
- **Feature Importance**: Some relevant features may not have been effectively utilized or appropriately scaled, affecting the model's ability to learn from the data.


### Next Steps

To address these issues, we plan to:

1. **Hyperparameter Tuning**: Implement systematic hyperparameter tuning using techniques such as Grid Search or Random Search to identify better configurations for the model.

2. **Cross-Validation**: Utilize k-fold cross-validation to ensure that the model's performance is robust and not just suitable for a specific train-test split.

3. **Feature Engineering**: Review and enhance the feature selection and engineering process to ensure that the model has access to the most relevant data for making predictions.

4. **Model Evaluation**: Regularly evaluate the model using metrics such as accuracy, precision, recall, and F1-score to monitor improvements and ensure the model meets performance expectations.

Through these steps, we aim to improve the model's effectiveness in predicting employee attrition and derive valuable insights for management.

## Installation Instructions

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Quang10A3ChuyenBo/Predict-employee-attrition-rate.git
   ```
2. Install the required libraries based on the provided requirements.txt:
   ```bash
   git clone https://github.com/Quang10A3ChuyenBo/Predict-employee-attrition-rate.git
   ```
3. Open Preprocess.ipynb to start preprocessing the data.

### Usage
- Use Training.ipynb to train the predictive models.
- Use Evaluation.ipynb to evaluate the performance of the trained models.
- *But to make it shorter, we have integrated the training, saving and loading model .pkl file in the Evaluation.ipynb* .

#### Contact us - Authors
- **Quang10A3ChuyenBo** :
- **NguyenHoangTu3241** :

## License
This project is licensed under the MIT License - see the LICENSE file for details.
