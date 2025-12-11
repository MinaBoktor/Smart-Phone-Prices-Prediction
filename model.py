import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from helper import preprocess
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    train = pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train.csv")
    test = pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\test.csv")

    train_preprocessed, training_columns, imputation_values, scaler = preprocess(train, train=True)
    test_preprocessed, _, _, _ = preprocess(test, train=False, training_columns=training_columns, imputation_values=imputation_values, scaler=scaler)

    train_preprocessed.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\preproccessed_train.csv", index=False)
    test_preprocessed.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\preproccessed_test.csv", index=False)

    # For training
    X_train = train_preprocessed.drop(["price"], axis=1)
    y_train = train_preprocessed["price"]

    # For testing
    X_test = test_preprocessed.drop("price", axis=1)
    y_test = test_preprocessed["price"]

    model = logistic(X_train, y_train)
    print("Logistic Regression Results:")
    evaluate(model, X_test, y_test)

    model = SVM(X_train, y_train)
    print("SVM Results:")
    evaluate(model, X_test, y_test)

    model = KNN(X_train, y_train)
    print("KNN Results:")
    evaluate(model, X_test, y_test)

    model = random_forest(X_train, y_train)
    print("Random Forest Results:")
    evaluate(model, X_test, y_test)

    model = decision_tree(X_train, y_train)
    print("Decision Tree Results:")
    evaluate(model, X_test, y_test)


def logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=0, solver='liblinear')
    model.fit(X_train, y_train)
    return model


def SVM(X_train, y_train):
    model = SVC(
        kernel='poly',
        C=1.0,
        gamma='scale',
        probability=True
    )

    model.fit(X_train, y_train)
    return model


def KNN(X_train, y_train):
    model = KNeighborsClassifier(
            n_neighbors=3,
            weights='uniform',
            metric='manhattan',
            algorithm='auto'
        )

    model.fit(X_train, y_train)
    return model

def random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        criterion='entropy',
        bootstrap=False,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=20, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        criterion='entropy', 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred)*100)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()