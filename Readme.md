Employee Attrition Analysis and Prediction
This project analyzes employee attrition data and builds a machine learning model to predict whether an employee is likely to leave the organization. It includes exploratory data analysis (EDA) and a Random Forest-based classification model.

Table of Contents
Overview
Features
Technologies Used
Getting Started
Running the Code
Jupyter Notebook Option
Results
License
Overview
The project performs the following:

Exploratory Data Analysis (EDA) using visualizations to uncover trends in the data.
Preprocessing of categorical and numerical features.
Implementation of a Random Forest Classifier for predicting employee attrition.
Evaluation of the model's performance on training and testing data.
Features
EDA Highlights:
Visualization of attrition rates by department, education, job role, and gender.
Age distribution plots.
Heatmap of correlations between numerical features.
Machine Learning:
Random Forest Classifier with hyperparameters:
n_estimators: 70
criterion: 'entropy'
max_depth: 20
Technologies Used
Python 3.7+
Libraries:
numpy
pandas
matplotlib
seaborn
scikit-learn
Getting Started
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-repo-link/Employee-Attrition.git
cd Employee-Attrition
Install Python and the required libraries:

bash
Copy
Edit
pip install numpy pandas matplotlib seaborn scikit-learn
Make sure the dataset (Dataset.csv) is in the same directory as the code files.

Running the Code
Option 1: Run the Python Script
Execute the Python script to view the analytics and train the model:

bash
Copy
Edit
python EDA+MODEL.py
This will:

Display the EDA graphs one by one.
Train the Random Forest model.
Print the training and testing accuracy.
Option 2: Use the Jupyter Notebook
Open the EDA+MODEL.ipynb file in Jupyter Notebook to:

View the code and output together in a well-structured format.
Save time by accessing pre-generated outputs for each cell.
Results
Exploratory Data Analysis: Insights into attrition rates by different features.
Model Performance:
Training Accuracy: ~95%
Testing Accuracy: ~90% (values may vary slightly based on the random seed).
License
This project is open-source and available under the MIT License.
