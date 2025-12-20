import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from helper import preprocess
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance


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
    print(f"KNN Results:")
    evaluate(model, X_test, y_test)

    visualize_knn_importance(model, X_test, y_test)

    #visualize_camera_vs_price(train)


    model = random_forest(X_train, y_train)
    print("Random Forest Results:")
    evaluate(model, X_test, y_test)

    model = decision_tree(X_train, y_train)
    print("Decision Tree Results:")
    evaluate(model, X_test, y_test)

    model = xgboost_model(X_train, y_train)
    print("Xgboost Results:")
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


def KNN(X_train, y_train, metric='manhattan'):
    model = KNeighborsClassifier(
            n_neighbors=3,
            weights='uniform',
            metric=metric,
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

def xgboost_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=1.5,  # A gentler balance than 'balanced'
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred)*100)
    #print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    #print("Classification Report:\n", classification_report(y_test, y_pred))


def visualize_knn_importance(model, X_test, y_test):
    print("Calculating KNN Permutation Importance... (this may take a moment)")
    
    # Calculate permutation importance
    results = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    # Create a DataFrame
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': results.importances_mean
    })
    
    # Sort by importance and TAKE ONLY THE TOP 20
    top_20 = importance_df.sort_values(by='Importance', ascending=False).head(20)
    
    # Plot
    plt.figure(figsize=(10, 8))  # Increased height slightly
    sns.barplot(x='Importance', y='Feature', data=top_20, palette='viridis')
    plt.title('Top 20 Features driving Price (KNN)')
    plt.xlabel('Importance (Drop in Accuracy)')
    plt.tight_layout()
    plt.show()




    # Print the text version too, just in case
    print("\n--- Top 10 Features ---")
    print(top_20.head(10))




def visualize_camera_vs_price(df):
    plt.figure(figsize=(10, 6))
    
    # We use a boxplot to see the price range for each Camera MP count
    sns.boxplot(x='primary_front_camera_mp', y='price', data=df)
    
    plt.title('Does a Better Selfie Camera Mean a Higher Price?')
    plt.xlabel('Front Camera Megapixels')
    plt.ylabel('Price Category')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()