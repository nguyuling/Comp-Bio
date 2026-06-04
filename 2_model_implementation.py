"""
    NAME: NGU YU LING
    MATRIC NO: A23CS0149
"""

#! 2. Model Implementation

# data loading
import pandas as pd
data = pd.read_csv('data.csv')

# assign features & target
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# model testing
y_pred = rfc.predict(X_test)