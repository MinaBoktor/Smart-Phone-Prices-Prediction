import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os

# Scikit-learn & Models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector

# Import your local helper
try:
    from helper import preprocess
except ImportError:
    st.error("Error: 'helper.py' not found. Please ensure app.py is in the same folder as helper.py")

warnings.filterwarnings('ignore')

# --- Utility: Parse User Inputs ---
def parse_list(input_str, type_func=float):
    """Parses '10, 50, 100' into [10, 50, 100]"""
    if not input_str: return None
    try:
        return [type_func(x.strip()) for x in input_str.split(',')]
    except:
        return []

def main():
    st.set_page_config(page_title="Jamaica", layout="wide",page_icon="📱")
    st.title("Jamaica Inc (Smartphone Price Prediction)")

    # ==========================================
    # 1. DATA LOADING
    # ==========================================
    st.sidebar.header("1. Data Source")
    
    # Default path from your provided code
    default_dir = r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction"
    
    data_mode = st.sidebar.radio("Load Mode", ["Use Local Path", "Upload CSVs"])
    
    train_df = None
    test_df = None

    if data_mode == "Use Local Path":
        project_path = st.sidebar.text_input("Project Path", default_dir)
        if st.sidebar.button("Load Data from Path"):
            try:
                t_path = os.path.join(project_path, "train.csv")
                e_path = os.path.join(project_path, "test.csv")
                train_df = pd.read_csv(t_path)
                test_df = pd.read_csv(e_path)
                st.session_state['data_loaded'] = True
                st.session_state['train_raw'] = train_df
                st.session_state['test_raw'] = test_df
                st.sidebar.success(f"Loaded: {len(train_df)} training samples.")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
    else:
        u_train = st.sidebar.file_uploader("Upload train.csv", type='csv')
        u_test = st.sidebar.file_uploader("Upload test.csv", type='csv')
        if u_train and u_test:
            train_df = pd.read_csv(u_train)
            test_df = pd.read_csv(u_test)
            st.session_state['data_loaded'] = True
            st.session_state['train_raw'] = train_df
            st.session_state['test_raw'] = test_df

    # ==========================================
    # 2. PREPROCESSING & FEATURE SELECTION
    # ==========================================
    if st.session_state.get('data_loaded'):
        st.subheader("2. Preprocessing & Features")
        
        col_p1, col_p2 = st.columns([1, 2])
        
        with col_p1:
            st.info("Clean & Encode")
            if st.button("Run Preprocessing (helper.py)"):
                with st.spinner("Processing..."):
                    # Using your helper.preprocess logic
                    tr_raw = st.session_state['train_raw']
                    te_raw = st.session_state['test_raw']
                    
                    tr_pre, cols, imp_vals, scaler, artifacts = preprocess(tr_raw, train=True)
                    te_pre, _, _, _, _ = preprocess(te_raw, train=False, 
                                                    training_columns=cols, 
                                                    imputation_values=imp_vals, 
                                                    scaler=scaler, 
                                                    imputer_artifacts=artifacts)
                    
                    st.session_state['X_train'] = tr_pre.drop(["price"], axis=1)
                    st.session_state['y_train'] = tr_pre["price"]
                    st.session_state['X_test'] = te_pre.drop("price", axis=1)
                    st.session_state['y_test'] = te_pre["price"]
                    st.session_state['preprocessed'] = True
                    st.success("Data Preprocessed!")

        if st.session_state.get('preprocessed'):
            with col_p2:
                st.info("Feature Selection (SFS)")
                n_feats = st.slider("Target Number of Features", 5, 50, 30)
                
                if st.button("Run SFS (Feature Selection)"):
                    with st.spinner("Running Sequential Feature Selection..."):
                        # Lightweight RF for selection (as per your code)
                        rf_fast = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=-1, random_state=42)
                        sfs = SequentialFeatureSelector(rf_fast, n_features_to_select=n_feats, direction='forward', scoring='accuracy', cv=3, n_jobs=-1)
                        
                        sfs.fit(st.session_state['X_train'], st.session_state['y_train'])
                        selected = list(sfs.get_feature_names_out(input_features=st.session_state['X_train'].columns))
                        
                        st.session_state['selected_features'] = selected
                        st.success(f"Selected {len(selected)} Features!")
                        st.write(selected)

    # ==========================================
    # 3. MODEL CONFIGURATION
    # ==========================================
    if st.session_state.get('preprocessed'):
        st.markdown("---")
        st.subheader("3. Model Configuration & Training")
        
        # Prepare Data
        X_tr = st.session_state['X_train']
        X_te = st.session_state['X_test']
        
        # Filter by selected features if they exist
        if 'selected_features' in st.session_state:
            X_tr = X_tr[st.session_state['selected_features']]
            X_te = X_te[st.session_state['selected_features']]
            st.caption(f"Training on {X_tr.shape[1]} features.")
        else:
            st.caption(f"Training on ALL {X_tr.shape[1]} features (SFS not run).")

        model_type = st.selectbox("Select Model Architecture", 
                                  ["Logistic Regression", "SVM", "KNN", "Random Forest", "XGBoost"])

        # Dynamic Hyperparameters
        params = {}
        st.write("#### Hyperparameter Grid")
        c1, c2 = st.columns(2)

        if model_type == "Logistic Regression":
            with c1:
                st.markdown("**C (Inverse Regularization)**")
                c_in = st.text_input("Values (comma sep)", "0.001, 0.01, 0.1, 1, 10, 100")
                params['C'] = parse_list(c_in)
                
                st.markdown("**Penalty**")
                params['penalty'] = st.multiselect("Types", ['l1', 'l2', 'elasticnet'], ['l2'])
            with c2:
                st.markdown("**Solver**")
                params['solver'] = st.multiselect("Types", ['liblinear', 'lbfgs', 'newton-cg'], ['lbfgs'])
                st.markdown("**Max Iterations**")
                params['max_iter'] = [st.number_input("Max Iter", 100, 5000, 2000)]
            
            estimator = LogisticRegression(random_state=0)

        elif model_type == "SVM":
            with c1:
                st.markdown("**C**")
                params['C'] = parse_list(st.text_input("C values", "0.1, 1, 10, 100"))
                st.markdown("**Kernel**")
                params['kernel'] = st.multiselect("Kernels", ['linear', 'rbf', 'poly', 'sigmoid'], ['rbf'])
            with c2:
                st.markdown("**Gamma**")
                g_in = st.text_input("Gamma (comma sep or 'scale')", "scale, auto")
                # Handle gamma parsing (can be float or string)
                g_vals = []
                for x in g_in.split(','):
                    x = x.strip()
                    try: g_vals.append(float(x))
                    except: g_vals.append(x)
                params['gamma'] = g_vals
            
            estimator = SVC(probability=True)

        elif model_type == "KNN":
            with c1:
                st.markdown("**N Neighbors**")
                params['n_neighbors'] = parse_list(st.text_input("Values", "3, 5, 7, 11, 15"), int)
                st.markdown("**Weights**")
                params['weights'] = st.multiselect("Type", ['uniform', 'distance'], ['uniform', 'distance'])
            with c2:
                st.markdown("**Metric**")
                params['metric'] = st.multiselect("Type", ['euclidean', 'manhattan'], ['euclidean', 'manhattan'])
            
            estimator = KNeighborsClassifier()

        elif model_type == "Random Forest":
            with c1:
                st.markdown("**N Estimators**")
                params['n_estimators'] = parse_list(st.text_input("Values", "50, 100, 200"), int)
                st.markdown("**Max Depth**")
                d_in = st.text_input("Values (0 for None)", "10, 20, 0")
                depths = parse_list(d_in, int)
                params['max_depth'] = [None if x == 0 else x for x in depths]
            with c2:
                st.markdown("**Criterion**")
                params['criterion'] = st.multiselect("Type", ['gini'], ['gini'])
            
            estimator = RandomForestClassifier(random_state=42)

        elif model_type == "XGBoost":
            with c1:
                st.markdown("**N Estimators**")
                params['n_estimators'] = parse_list(st.text_input("Values", "100, 300"), int)
                st.markdown("**Learning Rate**")
                params['learning_rate'] = parse_list(st.text_input("Values", "0.01, 0.05, 0.1"))
            with c2:
                st.markdown("**Max Depth**")
                params['max_depth'] = parse_list(st.text_input("Values", "3, 6, 10"), int)
            
            estimator = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)

        # ==========================================
        # 4. EXECUTION
        # ==========================================
        st.markdown("---")
        cv = st.slider("Cross Validation Folds (CV)", 2, 10, 5)
        
        if st.button("🚀 Start Grid Search"):
            st.write(f"Training {model_type}...")
            progress = st.progress(0)
            
            try:
                grid = GridSearchCV(estimator, params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
                
                with st.spinner("Fitting models... (Check terminal for details)"):
                    grid.fit(X_tr, st.session_state['y_train'])
                
                progress.progress(100)
                
                # Best Params
                st.success("Training Complete!")
                st.json(grid.best_params_)
                
                # Metrics
                train_acc = grid.best_score_ * 100
                y_pred = grid.best_estimator_.predict(X_te)
                test_acc = accuracy_score(st.session_state['y_test'], y_pred) * 100
                
                m1, m2 = st.columns(2)
                m1.metric("Validation Accuracy (CV)", f"{train_acc:.2f}%")
                m2.metric("Test Set Accuracy", f"{test_acc:.2f}%")
                
            except Exception as e:
                st.error(f"Training Failed: {e}")
                st.warning("Check your parameter combinations (e.g., Solver vs Penalty compatibility in Logistic Regression).")

if __name__ == "__main__":
    main()