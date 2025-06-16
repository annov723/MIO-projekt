import numpy as np
import pandas as pd
from mealpy import FloatVar
from mealpy.swarm_based.GWO import OriginalGWO
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from helper import remove_outliers_turkey, display_metrics, train_and_evaluate_genfis


def create_seeds_fuzzy_system(params=None):
    area = ctrl.Antecedent(np.arange(10, 25, 0.1), 'area')
    perimeter = ctrl.Antecedent(np.arange(12, 18, 0.1), 'perimeter')
    compactness = ctrl.Antecedent(np.arange(0.80, 0.92, 0.001), 'compactness')
    length = ctrl.Antecedent(np.arange(4.5, 7, 0.1), 'length')
    width = ctrl.Antecedent(np.arange(2.5, 4.5, 0.1), 'width')
    asymmetry = ctrl.Antecedent(np.arange(0, 8, 0.1), 'asymmetry')
    groove = ctrl.Antecedent(np.arange(4, 7, 0.1), 'groove')

    seed_class = ctrl.Consequent(np.arange(1, 4, 1), 'seed_class')

    if params is not None:
        p_area = np.sort(params[0:9])
        p_perim = np.sort(params[9:18])
        p_comp = np.sort(params[18:27])
        p_len = np.sort(params[27:36])
        p_width = np.sort(params[36:45])
        p_asym = np.sort(params[45:54])
        p_groove = np.sort(params[54:63])

        area['small'] = fuzz.trimf(area.universe, [p_area[0], p_area[1], p_area[2]])
        area['medium'] = fuzz.trimf(area.universe, [p_area[3], p_area[4], p_area[5]])
        area['large'] = fuzz.trimf(area.universe, [p_area[6], p_area[7], p_area[8]])
        perimeter['small'] = fuzz.trimf(perimeter.universe, [p_perim[0], p_perim[1], p_perim[2]])
        perimeter['medium'] = fuzz.trimf(perimeter.universe, [p_perim[3], p_perim[4], p_perim[5]])
        perimeter['large'] = fuzz.trimf(perimeter.universe, [p_perim[6], p_perim[7], p_perim[8]])
        compactness['low'] = fuzz.trimf(compactness.universe, [p_comp[0], p_comp[1], p_comp[2]])
        compactness['medium'] = fuzz.trimf(compactness.universe, [p_comp[3], p_comp[4], p_comp[5]])
        compactness['high'] = fuzz.trimf(compactness.universe, [p_comp[6], p_comp[7], p_comp[8]])
        length['short'] = fuzz.trimf(length.universe, [p_len[0], p_len[1], p_len[2]])
        length['medium'] = fuzz.trimf(length.universe, [p_len[3], p_len[4], p_len[5]])
        length['long'] = fuzz.trimf(length.universe, [p_len[6], p_len[7], p_len[8]])
        width['narrow'] = fuzz.trimf(width.universe, [p_width[0], p_width[1], p_width[2]])
        width['medium'] = fuzz.trimf(width.universe, [p_width[3], p_width[4], p_width[5]])
        width['wide'] = fuzz.trimf(width.universe, [p_width[6], p_width[7], p_width[8]])
        asymmetry['low'] = fuzz.trimf(asymmetry.universe, [p_asym[0], p_asym[1], p_asym[2]])
        asymmetry['medium'] = fuzz.trimf(asymmetry.universe, [p_asym[3], p_asym[4], p_asym[5]])
        asymmetry['high'] = fuzz.trimf(asymmetry.universe, [p_asym[6], p_asym[7], p_asym[8]])
        groove['short'] = fuzz.trimf(groove.universe, [p_groove[0], p_groove[1], p_groove[2]])
        groove['medium'] = fuzz.trimf(groove.universe, [p_groove[3], p_groove[4], p_groove[5]])
        groove['long'] = fuzz.trimf(groove.universe, [p_groove[6], p_groove[7], p_groove[8]])

    else:
        area['small'] = fuzz.trimf(area.universe, [10, 12, 15])
        area['medium'] = fuzz.trimf(area.universe, [13, 16, 19])
        area['large'] = fuzz.trimf(area.universe, [17, 20, 25])
        perimeter['small'] = fuzz.trimf(perimeter.universe, [12, 13, 14.5])
        perimeter['medium'] = fuzz.trimf(perimeter.universe, [14, 15, 16])
        perimeter['large'] = fuzz.trimf(perimeter.universe, [15.5, 16.5, 18])
        compactness['low'] = fuzz.trimf(compactness.universe, [0.80, 0.83, 0.86])
        compactness['medium'] = fuzz.trimf(compactness.universe, [0.85, 0.87, 0.89])
        compactness['high'] = fuzz.trimf(compactness.universe, [0.88, 0.90, 0.92])
        length['short'] = fuzz.trimf(length.universe, [4.5, 5.0, 5.5])
        length['medium'] = fuzz.trimf(length.universe, [5.3, 5.8, 6.3])
        length['long'] = fuzz.trimf(length.universe, [6.0, 6.5, 7.0])
        width['narrow'] = fuzz.trimf(width.universe, [2.5, 3.0, 3.4])
        width['medium'] = fuzz.trimf(width.universe, [3.2, 3.5, 3.8])
        width['wide'] = fuzz.trimf(width.universe, [3.7, 4.0, 4.5])
        asymmetry['low'] = fuzz.trimf(asymmetry.universe, [0, 1.5, 3])
        asymmetry['medium'] = fuzz.trimf(asymmetry.universe, [2.5, 4, 5.5])
        asymmetry['high'] = fuzz.trimf(asymmetry.universe, [5, 6, 8])
        groove['short'] = fuzz.trimf(groove.universe, [4, 4.5, 5])
        groove['medium'] = fuzz.trimf(groove.universe, [4.9, 5.4, 5.9])
        groove['long'] = fuzz.trimf(groove.universe, [5.7, 6.2, 7])

    seed_class.defuzzify_method = 'centroid'
    seed_class['class_1'] = fuzz.trimf(seed_class.universe, [1, 1, 2])
    seed_class['class_2'] = fuzz.trimf(seed_class.universe, [1, 2, 3])
    seed_class['class_3'] = fuzz.trimf(seed_class.universe, [2, 3, 3])

    rule1 = ctrl.Rule(area['large'] | width['wide'] | length['long'], seed_class['class_2'])
    rule2 = ctrl.Rule(groove['long'] | perimeter['large'], seed_class['class_2'])
    rule3 = ctrl.Rule(compactness['medium'] | area['medium'], seed_class['class_1'])
    rule4 = ctrl.Rule(asymmetry['high'] | area['small'], seed_class['class_3'])
    rule5 = ctrl.Rule(groove['short'] | compactness['high'], seed_class['class_3'])

    seeds_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    return ctrl.ControlSystemSimulation(seeds_ctrl)

def predict_for_seeds(fuzzy_system, data):
    DEFAULT_PREDICTION = 1

    predictions = []
    for row in data:
        try:
            fuzzy_system.input['area'] = row[0]
            fuzzy_system.input['perimeter'] = row[1]
            fuzzy_system.input['compactness'] = row[2]
            fuzzy_system.input['length'] = row[3]
            fuzzy_system.input['width'] = row[4]
            fuzzy_system.input['asymmetry'] = row[5]
            fuzzy_system.input['groove'] = row[6]
            fuzzy_system.compute()
            predicted_value = fuzzy_system.output.get('seeds_class')

            if predicted_value is None:
                predictions.append(DEFAULT_PREDICTION)
            else:
                predicted_class = int(round(predicted_value))
                predictions.append(predicted_class)
        except Exception as e:
            print(f'Error: {e}')
            predictions.append(DEFAULT_PREDICTION)
    return np.array(predictions)

def seeds_model_cv(X_seeds, y_seeds, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    original_scores = []
    optimized_scores = []

    for fold, (train_index, test_index) in enumerate(kfold.split(X_seeds, y_seeds)):
        print(f"\n===== FOLD {fold + 1}/{n_splits} =====\n")
        X_train, X_test = X_seeds[train_index], X_seeds[test_index]
        y_train, y_test = y_seeds[train_index], y_seeds[test_index]

        original_fuzzy_system = create_seeds_fuzzy_system(params=None)
        y_pred_original = predict_for_seeds(original_fuzzy_system, X_test)
        accuracy_original = accuracy_score(y_test, y_pred_original)
        original_scores.append(accuracy_original)
        print(f"FOLD {fold + 1} >> Accuracy ORIGINAL Fuzzy System: {accuracy_original:.4f}")

        def objective_function_seeds(solution):
            temp_fuzzy_system = create_seeds_fuzzy_system(params=solution)
            y_pred = predict_for_seeds(temp_fuzzy_system, X_train)
            return 1.0 - accuracy_score(y_train, y_pred)

        problem_dict = {
            "obj_func": objective_function_seeds,
            "bounds": FloatVar(
                lb=[
                    10, 11, 13, 12, 14, 17, 16, 18, 22,  # Area
                    12, 12.5, 13.5, 13, 14.5, 15.5, 15, 16, 17,  # Perimeter
                    0.80, 0.82, 0.84, 0.83, 0.86, 0.88, 0.87, 0.89, 0.91,  # Compactness
                    4.5, 4.8, 5.2, 5.0, 5.5, 6.0, 5.8, 6.2, 6.8,  # Length
                    2.5, 2.8, 3.1, 3.0, 3.3, 3.6, 3.5, 3.8, 4.2,  # Width
                    0, 1, 2.5, 2, 3.5, 5, 4.5, 5.5, 7.5,  # Asymmetry
                    4, 4.2, 4.8, 4.5, 5.2, 5.7, 5.5, 6.0, 6.8,  # Groove
                ],
                ub=[
                    13, 14, 16, 17, 18, 20, 22, 24, 25,  # Area
                    13.5, 14, 15, 15.5, 16, 16.5, 17, 17.5, 18,  # Perimeter
                    0.84, 0.85, 0.87, 0.88, 0.89, 0.90, 0.90, 0.91, 0.92,  # Compactness
                    5.2, 5.4, 5.8, 6.0, 6.2, 6.5, 6.8, 6.9, 7.0,  # Length
                    3.1, 3.3, 3.5, 3.6, 3.8, 4.0, 4.2, 4.4, 4.5,  # Width
                    2.5, 3, 4.5, 5, 5.5, 6.5, 7.5, 7.8, 8.0,  # Asymmetry
                    4.8, 5.0, 5.5, 5.7, 5.9, 6.2, 6.8, 6.9, 7.0,  # Groove
                ]),
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        }

        optimizer = OriginalGWO(epoch=50, pop_size=30)
        result = optimizer.solve(problem_dict)
        best_solution = result.solution

        optimized_fuzzy_system = create_seeds_fuzzy_system(params=best_solution)
        y_pred_optimized = predict_for_seeds(optimized_fuzzy_system, X_test)
        accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
        optimized_scores.append(accuracy_optimized)
        print(f"FOLD {fold + 1} >> Accuracy GWO-OPTIMIZED Fuzzy System: {accuracy_optimized:.4f}")

    print("\n\n--- CROSS-VALIDATION FINAL RESULTS ---\n")
    print(f"Original model accuracy: {np.mean(original_scores):.4f} +/- {np.std(original_scores):.4f}")
    print(f"GWO-Optimized model accuracy: {np.mean(optimized_scores):.4f} +/- {np.std(optimized_scores):.4f}")

def seeds_model():
    # Pre-processing SEEDS
    seeds_col_names = ['area', 'perimeter', 'compactness', 'length_of_kernel', 'width_of_kernel',
                       'asymmetry_coefficient', 'length_of_kernel_groove', 'class']
    seeds_data = pd.read_csv('data/seeds_dataset.txt', sep=r'\s+', header=None, engine='python',
                             names=seeds_col_names)
    seeds_data.dropna(inplace=True)
    X_seeds = seeds_data.iloc[:, :-1].values
    y_seeds = seeds_data.iloc[:, -1].values.astype(int)

    X_seeds_no_outliers, y_seeds_no_outliers = remove_outliers_turkey(X_seeds, y_seeds)

    print("\nEtykiety (y_seeds):")
    print(np.unique(y_seeds))
    print("Rozmiar przed eliminacją wartości skrajnych:", X_seeds.shape)
    print("Rozmiar po eliminacji wartości skrajnych:", X_seeds_no_outliers.shape)

    X_seeds = X_seeds_no_outliers
    y_seeds = y_seeds_no_outliers

    # Trenowanie i testowanie SEEDS
    X_train, X_test, y_train, y_test = train_test_split(
        X_seeds, y_seeds, test_size=0.2, random_state=42, stratify=y_seeds
    )

    print("\n\n\nPodział danych SEEDS:")
    print(f"Training: {X_train.shape[0]}")
    print(f"Test: {X_test.shape[0]}")

    original_fuzzy_system = create_seeds_fuzzy_system(params=None)
    y_pred_original = predict_for_seeds(original_fuzzy_system, X_test)

    def objective_function_seeds(solution):
        temp_fuzzy_system = create_seeds_fuzzy_system(params=solution)
        y_pred = predict_for_seeds(temp_fuzzy_system, X_train)
        return 1.0 - accuracy_score(y_train, y_pred)

    problem_dict = {
        "obj_func": objective_function_seeds,
        "bounds": FloatVar(
            lb=[
                10, 11, 13, 12, 14, 17, 16, 18, 22,  # Area
                12, 12.5, 13.5, 13, 14.5, 15.5, 15, 16, 17,  # Perimeter
                0.80, 0.82, 0.84, 0.83, 0.86, 0.88, 0.87, 0.89, 0.91,  # Compactness
                4.5, 4.8, 5.2, 5.0, 5.5, 6.0, 5.8, 6.2, 6.8,  # Length
                2.5, 2.8, 3.1, 3.0, 3.3, 3.6, 3.5, 3.8, 4.2,  # Width
                0, 1, 2.5, 2, 3.5, 5, 4.5, 5.5, 7.5,  # Asymmetry
                4, 4.2, 4.8, 4.5, 5.2, 5.7, 5.5, 6.0, 6.8,  # Groove
            ],
            ub=[
                13, 14, 16, 17, 18, 20, 22, 24, 25,  # Area
                13.5, 14, 15, 15.5, 16, 16.5, 17, 17.5, 18,  # Perimeter
                0.84, 0.85, 0.87, 0.88, 0.89, 0.90, 0.90, 0.91, 0.92,  # Compactness
                5.2, 5.4, 5.8, 6.0, 6.2, 6.5, 6.8, 6.9, 7.0,  # Length
                3.1, 3.3, 3.5, 3.6, 3.8, 4.0, 4.2, 4.4, 4.5,  # Width
                2.5, 3, 4.5, 5, 5.5, 6.5, 7.5, 7.8, 8.0,  # Asymmetry
                4.8, 5.0, 5.5, 5.7, 5.9, 6.2, 6.8, 6.9, 7.0,  # Groove
            ]
        ),
        "minmax": "min",
        "log_to": "console",
        "save_population": False,
    }

    optimizer = OriginalGWO(epoch=50, pop_size=30)
    result = optimizer.solve(problem_dict)

    best_solution = result.solution
    best_fitness = result.target.fitness

    print(f"Zakończono dla SEEDS.")
    print(f"Best Fitness (Error Rate on Train Set): {best_fitness:.4f}")
    print(f"Best Parameters Found: \n{best_solution}\n")


    optimized_fuzzy_system = create_seeds_fuzzy_system(params=best_solution)
    y_pred_optimized = predict_for_seeds(optimized_fuzzy_system, X_test)

    display_metrics(y_test, y_pred_original, "ORIGINAL SEEDS")
    display_metrics(y_test, y_pred_optimized, "GWO-OPTIMIZED SEEDS")

    seeds_model_cv(X_seeds, y_seeds, n_splits=5)
    train_and_evaluate_genfis(X_seeds, y_seeds, n_splits=5)



if __name__ == '__main__':
    seeds_model()