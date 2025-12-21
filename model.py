import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from helper import preprocess
from sklearn.feature_selection import SequentialFeatureSelector


def main():
    path = r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction"

    train = pd.read_csv(fr"{path}\train.csv")
    test = pd.read_csv(fr"{path}\test.csv")

    train_preprocessed, training_columns, imputation_values, scaler, imputer_artifacts = preprocess(train, train=True)
    test_preprocessed, _, _, _, _ = preprocess(test, train=False, training_columns=training_columns, imputation_values=imputation_values, scaler=scaler, imputer_artifacts=imputer_artifacts)

    # For training
    X_train = train_preprocessed.drop(["price"], axis=1)
    y_train = train_preprocessed["price"]

    # For testing
    X_test = test_preprocessed.drop("price", axis=1)
    y_test = test_preprocessed["price"]

    best_cols = select_features(X_train, y_train, n_features=30)

    X_train = X_train[best_cols]
    X_test = X_test[best_cols]

    # --- Training and Tuning Models ---

    print("--- Logistic Regression ---")
    model = logistic(X_train, y_train)
    evaluate(model, X_test, y_test)

    print("\n--- SVM ---")
    model = svm(X_train, y_train)
    evaluate(model, X_test, y_test)

    print("\n--- KNN ---")
    model = knn(X_train, y_train)
    evaluate(model, X_test, y_test)

    print("\n--- Random Forest ---")
    model = random_forest(X_train, y_train)
    evaluate(model, X_test, y_test)

    print("\n--- XGBoost ---")
    model = xgboost(X_train, y_train)
    evaluate(model, X_test, y_test)


def logistic(X_train, y_train):
    param_grid = [
        {
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'max_iter': [2000]
        },
        {
            'solver': ['lbfgs', 'newton-cg'],
            'penalty': ['l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'max_iter': [2000]
        }
    ]

    grid_search = GridSearchCV(LogisticRegression(random_state=0), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Params (Logistic): {grid_search.best_params_}")
    return grid_search.best_estimator_


def svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'degree': [2, 3, 4]
    }
    
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Params (SVM): {grid_search.best_params_}")
    return grid_search.best_estimator_


def knn(X_train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 300, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 10],
        'subsample': [0.8, 1.0]
    }

    xgb = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)

    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"Test Set Accuracy: {acc:.2f}%")


def select_features(X, y, n_features=30):

    print(f"\n--- Running Sequential Feature Selection (Target: {n_features} features) ---")
    print("Step 1: Initializing 'Lite' Random Forest...")

    rf_fast = RandomForestClassifier(
        n_estimators=10, 
        max_depth=5, 
        n_jobs=-1, 
        random_state=42
    )

    sfs = SequentialFeatureSelector(
        rf_fast,
        n_features_to_select=n_features,
        direction='forward',
        scoring='accuracy',
        cv=3, 
        n_jobs=4
    )

    print("Step 2: Fitting SFS (this usually takes 1-3 minutes)...")
    sfs.fit(X, y)

    selected_features = list(sfs.get_feature_names_out(input_features=X.columns))
    print(f"Selected Features: {selected_features}")
    return selected_features

if __name__ == "__main__":
    main()