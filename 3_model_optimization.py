#! 3. Model Optimization

# data loading
import pandas as pd
data = pd.read_csv('data.csv')

# assign features & target
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# construct an optimization pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('rfc', RandomForestClassifier(random_state=42))
])

# define the hyperparameter search space
param_grid = {
    'feature_selection__k': [20, 50, 100, 200], #top N most statistically significant genes
    'rfc__n_estimators': [50, 100, 200], # number of trees
    'rfc__max_depth': [None, 5, 10], # tree depth
    'rfc__min_samples_split': [2, 5], # minimum samples required to split a node
    'rfc__class_weight': ['balanced', 'balanced_subsample'] # adjusts for class imbalance (47 ALL vs 25 AML)
}

# 5-fold cross-validation
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

best_model = grid_search.best_estimator_

# model testing using the best model
y_pred = best_model.predict(X_test)

#! Before Optimization: Model Evaluation

# model performance scoring
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall: ', recall_score(y_test, y_pred, average='weighted'))
print('f1-score: ', f1_score(y_test, y_pred, average='weighted'))

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.figure(figsize=(6, 5))

import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True,
            xticklabels=sorted(y_test.unique()), 
            yticklabels=sorted(y_test.unique()),
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})

plt.title('Confusion Matrix: Optimized Pipeline', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=10)
plt.xlabel('Predicted Label', fontsize=10)
plt.savefig('confusion_matrix_after.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"{cm}\n")
print(f"True Negatives (TN):  {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP):  {cm[1,1]}")

# Optional: Print out the top selected genes to see what the model is looking at
selected_features_idx = best_model.named_steps['feature_selection'].get_support(indices=True)
top_genes = X.columns[selected_features_idx].tolist()
print(f"\nTop {len(top_genes)} Predictive Genes Selected (Subset):", top_genes[:10])