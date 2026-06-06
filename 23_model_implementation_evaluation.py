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

# visualization of first decision tree in RFC
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.figure(figsize=(25, 12))

from sklearn.tree import plot_tree
plot_tree(
    rfc.estimators_[0], 
    filled=True, 
    feature_names=list(X.columns), 
    class_names=[str(cls) for cls in sorted(y.unique())],
    rounded=True,
    max_depth=3 # Restricts depth to 3 levels so it remains completely readable
)

plt.tight_layout()
plt.savefig('first_decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()


#! 3. Evaluation & Short Discussioon

# model performance scoring
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall: ', recall_score(y_test, y_pred, average='weighted'))
print('f1: ', f1_score(y_test, y_pred, average='weighted'))

# confusion matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=sorted(y_test.unique()), 
            yticklabels=sorted(y_test.unique()),
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
plt.style.use('dark_background')
plt.title('Confusion Matrix of Random Forest Classifier', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n{cm}\n")
print(f"True Negatives (TN):  {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP):  {cm[1,1]}")

# discussion
"""
1. High Specificity: The model achieved perfect specificity (0% FP rate) for AML classification, meaning no AML samples were incorrectly classified as ALL.

2. Strong Overall Performance: The model achieved ~87% accuracy across both cancer types, indicating good generalization to unseen data.

3. Balanced Metrics: Precision and recall are closely aligned, suggesting the model is neither biased toward false positives nor false negatives.

4. Minor Classification Errors: 2 out of 15 test samples (13.3%) were misclassified, both false negatives (ALL samples predicted as AML). This suggests the model may be slightly conservative in predicting ALL.
"""