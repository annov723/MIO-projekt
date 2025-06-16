import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, \
    classification_report


def remove_outliers_turkey(X, y):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = (X >= lower_bound) & (X <= upper_bound)
    non_outlier_mask = np.all(mask, axis=1)
    return X[non_outlier_mask], y[non_outlier_mask]

def display_metrics(y_true, y_pred, title):
    print(f"--- {title} ---")

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix\n{cm}")
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\nDokładność (Accuracy): {accuracy:.4f}")
    print(f"Precyzja (Precision): {precision:.4f}")
    print(f"Czułość (Recall): {recall:.4f}")
    print(f"Miara F1 (F1-Score): {f1:.4f}\n")

    print("Raport klasyfikacji:")
    print(classification_report(y_true, y_pred, zero_division=0, target_names=['Klasa 1', 'Klasa 2', 'Klasa 3']))
