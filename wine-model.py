import numpy as np
import pandas as pd
from mealpy import FloatVar
from mealpy.swarm_based.GWO import OriginalGWO
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from skfuzzy import control as ctrl
import skfuzzy as fuzz

from helper import remove_outliers_turkey, display_metrics, train_and_evaluate_genfis


def create_wine_fuzzy_system(params=None):
    alcohol = ctrl.Antecedent(np.arange(11, 15, 0.1), 'alcohol')
    malic_acid = ctrl.Antecedent(np.arange(0.5, 6, 0.1), 'malic_acid')
    flavanoids = ctrl.Antecedent(np.arange(0.5, 6, 0.1), 'flavanoids')
    proline = ctrl.Antecedent(np.arange(200, 1700, 10), 'proline')

    wine_class = ctrl.Consequent(np.arange(1, 4, 0.1), 'wine_class')

    if params is not None:
        p_alc = params[:12]
        p_malic = params[12:21]
        p_flav = params[21:33]
        p_pro = params[33:45]

        alcohol['low'] = fuzz.trimf(alcohol.universe, sorted([p_alc[0], p_alc[1], p_alc[2]]))
        alcohol['medium'] = fuzz.trimf(alcohol.universe, sorted([p_alc[3], p_alc[4], p_alc[5]]))
        alcohol['high'] = fuzz.trimf(alcohol.universe, sorted([p_alc[6], p_alc[7], p_alc[8]]))
        alcohol['very_high'] = fuzz.trimf(alcohol.universe, sorted([p_alc[9], p_alc[10], p_alc[11]]))
        malic_acid['low'] = fuzz.trimf(malic_acid.universe, sorted([p_malic[0], p_malic[1], p_malic[2]]))
        malic_acid['medium'] = fuzz.trimf(malic_acid.universe, sorted([p_malic[3], p_malic[4], p_malic[5]]))
        malic_acid['high'] = fuzz.trimf(malic_acid.universe, sorted([p_malic[6], p_malic[7], p_malic[8]]))
        flavanoids['low'] = fuzz.trimf(flavanoids.universe, sorted([p_flav[0], p_flav[1], p_flav[2]]))
        flavanoids['medium'] = fuzz.trimf(flavanoids.universe, sorted([p_flav[3], p_flav[4], p_flav[5]]))
        flavanoids['high'] = fuzz.trimf(flavanoids.universe, sorted([p_flav[6], p_flav[7], p_flav[8]]))
        flavanoids['very_high'] = fuzz.trimf(flavanoids.universe, sorted([p_flav[9], p_flav[10], p_flav[11]]))
        proline['low'] = fuzz.trimf(proline.universe, sorted([p_pro[0], p_pro[1], p_pro[2]]))
        proline['medium'] = fuzz.trimf(proline.universe, sorted([p_pro[3], p_pro[4], p_pro[5]]))
        proline['high'] = fuzz.trimf(proline.universe, sorted([p_pro[6], p_pro[7], p_pro[8]]))
        proline['very_high'] = fuzz.trimf(proline.universe, sorted([p_pro[9], p_pro[10], p_pro[11]]))

    else:
        alcohol['low'] = fuzz.trimf(alcohol.universe, [11.0, 11.7, 12.5])
        alcohol['medium'] = fuzz.trimf(alcohol.universe, [12.3, 13.1, 13.7])
        alcohol['high'] = fuzz.trimf(alcohol.universe, [13.2, 13.7, 14.0])
        alcohol['very_high'] = fuzz.trimf(alcohol.universe, [13.8, 14.2, 14.5])
        malic_acid['low'] = fuzz.trimf(malic_acid.universe, [0.5, 1.3, 2.1])
        malic_acid['medium'] = fuzz.trimf(malic_acid.universe, [1.7, 2.4, 3.2])
        malic_acid['high'] = fuzz.trimf(malic_acid.universe, [2.8, 3.9, 5.5])
        flavanoids['low'] = fuzz.trimf(flavanoids.universe, [0.5, 1.1, 1.9])
        flavanoids['medium'] = fuzz.trimf(flavanoids.universe, [1.5, 2.3, 3.1])
        flavanoids['high'] = fuzz.trimf(flavanoids.universe, [2.7, 3.3, 4.0])
        flavanoids['very_high'] = fuzz.trimf(flavanoids.universe, [3.5, 4.3, 5.5])
        proline['low'] = fuzz.trimf(proline.universe, [200, 500, 900])
        proline['medium'] = fuzz.trimf(proline.universe, [700, 1100, 1400])
        proline['high'] = fuzz.trimf(proline.universe, [1100, 1300, 1500])
        proline['very_high'] = fuzz.trimf(proline.universe, [1400, 1550, 1700])

    wine_class.defuzzify_method = 'centroid'
    wine_class['class_1'] = fuzz.trimf(wine_class.universe, [1.0, 1.2, 1.8])
    wine_class['class_2'] = fuzz.trimf(wine_class.universe, [1.7, 2.0, 2.3])
    wine_class['class_3'] = fuzz.trimf(wine_class.universe, [2.2, 2.8, 3.0])

    rules = [
        ctrl.Rule(
            (alcohol['high'] | alcohol['very_high']) &
            (flavanoids['high'] | flavanoids['very_high']) &
            malic_acid['low'],
            wine_class['class_1']),
        ctrl.Rule(
            alcohol['very_high'] &
            flavanoids['high'] &
            (proline['medium'] | proline['high']),
            wine_class['class_1']),
        ctrl.Rule(
            wine_class['class_1'] &
            ~(malic_acid['medium'] | malic_acid['high']),
            wine_class['class_1']),
        ctrl.Rule(
            (flavanoids['low'] & (malic_acid['high'] | alcohol['low'])) |
            (proline['low'] & malic_acid['medium']),
            wine_class['class_3']),
        ctrl.Rule(flavanoids['high'] & alcohol['low'], wine_class['class_2']),
        ctrl.Rule(flavanoids['low'], wine_class['class_3']),
        ctrl.Rule(flavanoids['medium'], wine_class['class_2'])
    ]

    wine_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(wine_ctrl)

def predict_for_wine(fuzzy_system, data):
    DEFAULT_PREDICTION = 1

    predictions = []
    for sample in data:
        try:
            fuzzy_system.input['alcohol'] = sample[0]
            fuzzy_system.input['malic_acid'] = sample[1]
            fuzzy_system.input['flavanoids'] = sample[6]
            fuzzy_system.input['proline'] = sample[12]
            fuzzy_system.compute()
            predicted_value = fuzzy_system.output.get('wine_class')

            if predicted_value is None:
                predictions.append(DEFAULT_PREDICTION)
            else:
                predicted_class = int(round(predicted_value))
                predictions.append(predicted_class)
        except Exception as e:
            print(f'Error: {e}')
            predictions.append(DEFAULT_PREDICTION)
    return np.array(predictions)

def wine_model_cv(X_wine, y_wine, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    original_scores = []
    optimized_scores = []

    for fold, (train_index, test_index) in enumerate(kfold.split(X_wine, y_wine)):
        print(f"\n===== FOLD {fold + 1}/{n_splits} =====\n")
        X_train, X_test = X_wine[train_index], y_wine[train_index]
        y_train, y_test = X_wine[test_index], y_wine[test_index]

        original_fuzzy_system = create_wine_fuzzy_system(params=None)
        y_pred_original = predict_for_wine(original_fuzzy_system, y_train) # Note: predict on X_test (y_train here is a typo fix)
        accuracy_original = accuracy_score(y_test, y_pred_original)
        original_scores.append(accuracy_original)
        print(f"FOLD {fold + 1} >> Accuracy ORIGINAL Fuzzy System: {accuracy_original:.4f}")

        def objective_function_wine(solution):
            temp_fuzzy_system = create_wine_fuzzy_system(params=solution)
            y_pred = predict_for_wine(temp_fuzzy_system, X_train)
            return 1.0 - accuracy_score(y_train, y_pred)

        problem_dict = {
            "obj_func": objective_function_wine,
            "bounds": FloatVar(
                lb=[11.0, 11.5, 12.0, 12.0, 12.5, 13.0, 13.0, 13.5, 14.0, 13.5, 14.0, 14.2, 0.5, 1.0, 2.0, 1.5, 2.0,
                    3.0, 2.5, 3.5, 5.0, 1.0, 1.3, 2.0, 1.7, 2.0, 2.5, 2.2, 2.6, 3.2, 0.5, 1.0, 1.8, 1.3, 2.0, 3.0, 2.5,
                    3.0, 3.8, 3.2, 4.0, 5.0, 200, 400, 800, 600, 1000, 1300, 1000, 1200, 1400, 1300, 1500, 1600],
                ub=[11.5, 12.0, 13.0, 12.5, 13.5, 14.0, 13.5, 14.0, 14.5, 14.0, 14.5, 14.5, 1.0, 2.0, 3.0, 2.0, 3.0,
                    4.0, 3.5, 5.0, 6.0, 1.5, 2.0, 2.5, 2.0, 2.5, 3.0, 2.7, 3.0, 4.0, 1.0, 1.8, 2.5, 2.0, 3.0, 4.0, 3.0,
                    3.8, 5.0, 4.0, 5.0, 6.0, 400, 800, 1000, 800, 1300, 1500, 1200, 1400, 1700, 1500, 1700, 1700]
            ),
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        }

        optimizer = OriginalGWO(epoch=50, pop_size=30)
        result = optimizer.solve(problem_dict)
        best_solution = result.solution

        optimized_fuzzy_system = create_wine_fuzzy_system(params=best_solution)
        y_pred_optimized = predict_for_wine(optimized_fuzzy_system, y_train)
        accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
        optimized_scores.append(accuracy_optimized)
        print(f"FOLD {fold + 1} >> Accuracy GWO-OPTIMIZED Fuzzy System: {accuracy_optimized:.4f}")

    print("\n\n--- CROSS-VALIDATION FINAL RESULTS ---\n")
    print(f"Original model accuracy:    {np.mean(original_scores):.4f} ± {np.std(original_scores):.4f}")
    print(f"GWO-Optimized model accuracy: {np.mean(optimized_scores):.4f} ± {np.std(optimized_scores):.4f}")

def wine_model():
    # Pre-processing WINE
    wine_col_names = ['class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
                      'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue',
                      'OD280/OD315_of_diluted_wines', 'Proline']
    wine_data = pd.read_csv('data/wine.data', header=None, engine='python', names=wine_col_names)
    wine_data.dropna(inplace=True)
    X_wine = wine_data.iloc[:, 1:].values
    y_wine = wine_data.iloc[:, 0].values.astype(int)

    X_wine_no_outliers, y_wine_no_outliers = remove_outliers_turkey(X_wine, y_wine)

    print("\nEtykiety (y_wine):")
    print(np.unique(y_wine))
    print("Rozmiar przed eliminacją wartości skrajnych:", X_wine.shape)
    print("Rozmiar po eliminacji wartości skrajnych:", X_wine_no_outliers.shape)

    X_wine = X_wine_no_outliers
    y_wine = y_wine_no_outliers

    # Trenowanie i testowanie WINE
    X_train, X_test, y_train, y_test = train_test_split(
        X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
    )
    print("\n\n\nPodział danych WINE:")
    print(f"Training: {X_train.shape[0]}")
    print(f"Test: {X_test.shape[0]}")

    original_fuzzy_system = create_wine_fuzzy_system(params=None)
    y_pred_original = predict_for_wine(original_fuzzy_system, X_test)

    def objective_function_wine(solution):
        temp_fuzzy_system = create_wine_fuzzy_system(params=solution)
        y_pred = predict_for_wine(temp_fuzzy_system, X_train)
        return 1.0 - accuracy_score(y_train, y_pred)

    problem_dict = {
        "obj_func": objective_function_wine,
        "bounds": FloatVar(
            lb=[11.0, 11.5, 12.0, 12.0, 12.5, 13.0, 13.0, 13.5, 14.0, 13.5, 14.0, 14.2, 0.5, 1.0, 2.0, 1.5, 2.0, 3.0,
                2.5, 3.5, 5.0, 1.0, 1.3, 2.0, 1.7, 2.0, 2.5, 2.2, 2.6, 3.2, 0.5, 1.0, 1.8, 1.3, 2.0, 3.0, 2.5, 3.0, 3.8,
                3.2, 4.0, 5.0, 200, 400, 800, 600, 1000, 1300, 1000, 1200, 1400, 1300, 1500, 1600],
            ub=[11.5, 12.0, 13.0, 12.5, 13.5, 14.0, 13.5, 14.0, 14.5, 14.0, 14.5, 14.5, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0,
                3.5, 5.0, 6.0, 1.5, 2.0, 2.5, 2.0, 2.5, 3.0, 2.7, 3.0, 4.0, 1.0, 1.8, 2.5, 2.0, 3.0, 4.0, 3.0, 3.8, 5.0,
                4.0, 5.0, 6.0, 400, 800, 1000, 800, 1300, 1500, 1200, 1400, 1700, 1500, 1700, 1700]
        ),
        "minmax": "min",
        "log_to": "console",
        "save_population": False,
    }
    optimizer = OriginalGWO(epoch=50, pop_size=30)
    result = optimizer.solve(problem_dict)

    best_solution = result.solution
    best_fitness = result.target.fitness

    print(f"\nZakończono GWO dla WINE.")
    print(f"Best Fitness (Error Rate on Train Set): {best_fitness:.4f}")
    print(f"Najlepsze rozwiązanie: \n{best_solution}\n")

    optimized_fuzzy_system = create_wine_fuzzy_system(params=best_solution)
    y_pred_optimized = predict_for_wine(optimized_fuzzy_system, X_test)

    display_metrics(y_test, y_pred_original, "ORIGINAL WINE")
    display_metrics(y_test, y_pred_optimized, "GWO-OPTIMIZED WINE")

    wine_model_cv(X_wine, y_wine, n_splits=5)
    train_and_evaluate_genfis(X_wine, y_wine, n_splits=5)



if __name__ == '__main__':
    wine_model()