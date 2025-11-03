Titanic Survival Prediction using Random Forest

This project is a classic machine learning classification problem based on the Kaggle competition, "Titanic - Machine Learning from Disaster." The objective is to analyze the Titanic passenger dataset and build a predictive model to determine whether a passenger survived or not.

Dataset: Titanic - Machine Learning from Disaster

Project Pipeline

This notebook follows a structured machine learning workflow:

Exploratory Data Analysis (EDA):

Loaded the train.csv data using Pandas.

Investigated the features, data types, and missing values.

Performed visual analysis using Seaborn to find initial insights:

Survival by Gender: Confirmed that female passengers had a significantly higher survival rate than male passengers.

Survival by Class: Showed that 1st Class passengers had the highest survival rate, followed by 2nd and 3rd Class.

Data Preprocessing & Feature Engineering:

Missing Data Imputation:

Age: Missing values (177) were filled using the median age of all passengers.

Embarked: Missing values (2) were filled using the mode (most common embarkation port, 'S').

Cabin: This column was dropped as it was missing over 80% of its data.

Categorical Encoding:

The Sex column (male/female) was converted into a binary numeric feature (Sex_male) using pd.get_dummies.

New Feature Creation:

FamilySize: Created by combining SibSp (siblings/spouse) and Parch (parents/children) + 1.

isAlone: A binary feature derived from FamilySize (True if FamilySize == 1). This showed a negative correlation with survival, making it a useful feature.

Modeling and Evaluation:

Feature Selection: The model was trained on the following engineered features:

Pclass

Age

Fare

Sex_male

isAlone

Train/Test Split: The training data was split into an 80% training set and a 20% validation set.

Baseline Model (Logistic Regression):

A LogisticRegression model was trained as a baseline.

Validation Accuracy: 79.89%

Final Model (Random Forest):

A RandomForestClassifier (with n_estimators=100, max_depth=5) was trained, showing improved performance.

Final Validation Accuracy: 81.01%

How to Run

Download train.csv and test.csv from the official Kaggle competition page.

Ensure the notebook and data files are in the correct directory structure as expected by the notebook.

Install the required libraries: pandas, numpy, seaborn, matplotlib, scikit-learn.

Run the cells in the Jupyter Notebook sequentially.

Future Work

Correctly preprocess the test.csv file (impute missing Age/Fare and select the final features) to generate a submission.csv file for the Kaggle competition.

Experiment with other models (e.g., Gradient Boosting, XGBoost) to further improve accuracy.

Perform more advanced feature engineering, such as extracting titles from the Name column.
