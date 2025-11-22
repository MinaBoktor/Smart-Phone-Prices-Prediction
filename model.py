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


def main():
    train = pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train.csv")
    test = pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\test.csv")

    train, training_columns = preprocess(train, train=True)
    test, _ = preprocess(test, train=False, training_columns=training_columns)

    train.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\preproccessed_train.csv", index=False)
    test.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\preproccessed_test.csv", index=False)

    # For training
    X_train = train.drop(["price"], axis=1)
    y_train = train["price"]

    # For testing
    X_test = test.drop("price", axis=1)
    y_test = test["price"]

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
        kernel='poly',      # 'linear', 'poly', 'rbf', 'sigmoid'
        C=1.0,             # regularization parameter
        gamma='scale',     # kernel coefficient ('scale' is default)
        probability=True   # if you want predicted probabilities
    )

    model.fit(X_train, y_train)
    return model


def KNN(X_train, y_train):
    model = KNeighborsClassifier(
        n_neighbors=3,   # number of neighbors (tune this)
        weights='distance', # 'uniform' or 'distance'
        metric='minkowski', # distance metric ('euclidean' if p=2)
        p=2              # parameter for Minkowski (p=2 → Euclidean)
    )

    model.fit(X_train, y_train)
    return model

def random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=4, random_state=0)
    model.fit(X_train, y_train)
    return model

def decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=4, random_state=0)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred)*100)
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()