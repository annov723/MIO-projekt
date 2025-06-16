import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, \
    classification_report
from sklearn.model_selection import StratifiedKFold
import skfuzzy as fuzz


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
    print(f"\n--- {title} ---")

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

def train_and_evaluate_genfis(X, y, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    n_classes = len(np.unique(y))

    print(f"\n\n--- Evaluating GENFIS (Fuzzy C-Means) Model with {n_splits}-Fold CV ---\n")

    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        centers, u_train, _, _, _, _, _ = fuzz.cluster.cmeans(
            X_train.T, c=n_classes, m=2, error=0.005, maxiter=1000, init=None
        )

        cluster_labels = np.zeros(n_classes)
        train_labels_by_cluster = np.argmax(u_train, axis=0)
        for i in range(n_classes):
            points_in_cluster = y_train[train_labels_by_cluster == i]
            if len(points_in_cluster) > 0:
                cluster_labels[i] = np.bincount(points_in_cluster).argmax()
            else:
                cluster_labels[i] = np.random.choice(np.unique(y))

        u_test, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X_test.T, centers, m=2, error=0.005, maxiter=1000
        )

        predicted_clusters = np.argmax(u_test, axis=0)
        y_pred = np.array([cluster_labels[c] for c in predicted_clusters])

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"FOLD {fold + 1} >> Genfis Accuracy: {accuracy:.4f}")

    print("\n--- GENFIS FINAL RESULTS ---\n")
    print(f"Genfis (FCM) model accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
