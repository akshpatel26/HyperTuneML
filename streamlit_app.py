# Enhanced HyperTuneML Platform with New Features
# Added: Model Comparison, Performance Metrics Dashboard, Data Statistics, and Export Results

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import warnings
import time
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# NEW FEATURE 1: Model Performance Storage
class ModelPerformanceTracker:
    def __init__(self):
        if 'performance_history' not in st.session_state:
            st.session_state.performance_history = []
    
    def add_performance(self, model_name, dataset, algorithm_type, train_acc, test_acc, training_time):
        performance_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': model_name,
            'dataset': dataset,
            'type': algorithm_type,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'training_time': training_time
        }
        st.session_state.performance_history.append(performance_data)
    
    def get_history(self):
        return st.session_state.performance_history

# Initialize tracker
tracker = ModelPerformanceTracker()

def load_dataset(Data):
    if Data == "Iris":
        return datasets.load_iris()
    elif Data == "Wine":
        return datasets.load_wine()
    elif Data == "Breast Cancer":
        return datasets.load_breast_cancer()
    elif Data == "Diabetes":
        return datasets.load_diabetes()
    elif Data == "Digits":
        return datasets.load_digits()
    elif Data == "Salary":
        return pd.read_csv("Dataset/Salary_dataset.csv")
    elif Data == "Naive Bayes Classification":
        return pd.read_csv("Dataset/Naive-Bayes-Classification-Data.csv")
    elif Data == "Heart Disease Classification":
        return pd.read_csv("Dataset/Updated_heart_prediction.csv")
    elif Data == "Titanic":
        return pd.read_csv("Dataset/Preprocessed Titanic Dataset.csv")
    else:
        return pd.read_csv("Dataset/car_evaluation.csv")

def Input_output(data, data_name):
    if data_name == "Salary":
        X, Y = data["YearsExperience"].to_numpy().reshape(-1, 1), data["Salary"].to_numpy().reshape(-1, 1)
    elif data_name == "Naive Bayes Classification":
        X, Y = data.drop("diabetes", axis=1), data["diabetes"]
    elif data_name == "Heart Disease Classification":
        X, Y = data.drop("output", axis=1), data["output"]
    elif data_name == "Titanic":
        X, Y = (
            data.drop(columns=["survived", "home.dest", "last_name", "first_name", "title"], axis=1),
            data["survived"],
        )
    elif data_name == "Car Evaluation":
        df = data
        le = LabelEncoder()
        func = lambda i: le.fit(df[i]).transform(df[i])
        for i in df.columns:
            df[i] = func(i)
        X, Y = df.drop(["unacc"], axis=1), df["unacc"]
    else:
        X = data.data
        Y = data.target
    return X, Y

# NEW FEATURE 2: Enhanced Dataset Statistics
def show_dataset_statistics(data, data_name, X, Y):
    st.subheader("📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", X.shape[0])
    
    with col2:
        st.metric("Features", X.shape[1])
    
    with col3:
        if hasattr(data, 'target_names'):
            st.metric("Classes", len(np.unique(Y)))
        else:
            st.metric("Target Range", f"{Y.min():.2f} - {Y.max():.2f}")
    
    with col4:
        if len(np.unique(Y)) < 10:  # Classification
            class_counts = np.bincount(Y)
            balance_ratio = class_counts.min() / class_counts.max()
            st.metric("Class Balance", f"{balance_ratio:.2f}")

def add_parameter_classifier_general(algorithm):
    params = dict()
    
    if algorithm == "SVM":
        c_regular = st.sidebar.slider("C (Regularization)", 0.01, 10.0)
        kernel_custom = st.sidebar.selectbox("Kernel", ("linear", "poly ", "rbf", "sigmoid"))
        params["C"] = c_regular
        params["kernel"] = kernel_custom
    
    elif algorithm == "KNN":
        k_n = st.sidebar.slider("Number of Neighbors (K)", 1, 20, key="k_n_slider")
        params["K"] = k_n
        weights_custom = st.sidebar.selectbox("Weights", ("uniform", "distance"))
        params["weights"] = weights_custom
    
    elif algorithm == "Naive Bayes":
        st.sidebar.info("This is a simple Algorithm. It doesn't have Parameters for Hyper-tuning.")
    
    elif algorithm == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
        params["max_depth"] = max_depth
        params["criterion"] = criterion
        params["splitter"] = splitter
        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random)
        except:
            params["random_state"] = 4567
    
    elif algorithm == "Random Forest":
        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 90)
        criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy", "log_loss"))
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random)
        except:
            params["random_state"] = 4567
    
    else:  # Logistic Regression
        c_regular = st.sidebar.slider("C (Regularization)", 0.01, 10.0)
        params["C"] = c_regular
        fit_intercept = st.sidebar.selectbox("Fit Intercept", ("True", "False"))
        params["fit_intercept"] = bool(fit_intercept)
        penalty = st.sidebar.selectbox("Penalty", ("l2", None))
        params["penalty"] = penalty
        n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
        params["n_jobs"] = n_jobs
    
    return params

def add_parameter_regressor(algorithm):
    params = dict()
    
    if algorithm == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        criterion = st.sidebar.selectbox("Criterion", ("absolute_error", "squared_error", "poisson", "friedman_mse"))
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
        params["max_depth"] = max_depth
        params["criterion"] = criterion
        params["splitter"] = splitter
        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random)
        except:
            params["random_state"] = 4567
    
    elif algorithm == "Linear Regression":
        fit_intercept = st.sidebar.selectbox("Fit Intercept", ("True", "False"))
        params["fit_intercept"] = bool(fit_intercept)
        n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
        params["n_jobs"] = n_jobs
    
    else:  # Random Forest
        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 90)
        criterion = st.sidebar.selectbox("Criterion", ("absolute_error", "squared_error", "poisson", "friedman_mse"))
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random)
        except:
            params["random_state"] = 4567
    
    return params

def model_classifier(algorithm, params):
    if algorithm == "KNN":
        return KNeighborsClassifier(n_neighbors=params["K"], weights=params["weights"])
    elif algorithm == "SVM":
        return SVC(C=params["C"], kernel=params["kernel"])
    elif algorithm == "Decision Tree":
        return DecisionTreeClassifier(
            criterion=params["criterion"],
            splitter=params["splitter"],
            random_state=params["random_state"],
        )
    elif algorithm == "Naive Bayes":
        return GaussianNB()
    elif algorithm == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            criterion=params["criterion"],
            random_state=params["random_state"],
        )
    elif algorithm == "Linear Regression":
        return LinearRegression(fit_intercept=params["fit_intercept"], n_jobs=params["n_jobs"])
    else:
        return LogisticRegression(
            fit_intercept=params["fit_intercept"],
            penalty=params["penalty"],
            C=params["C"],
            n_jobs=params["n_jobs"],
        )

def model_regressor(algorithm, params):
    if algorithm == "KNN":
        return KNeighborsRegressor(n_neighbors=params["K"], weights=params["weights"])
    elif algorithm == "SVM":
        return SVR(C=params["C"], kernel=params["kernel"])
    elif algorithm == "Decision Tree":
        return DecisionTreeRegressor(
            criterion=params["criterion"],
            splitter=params["splitter"],
            random_state=params["random_state"],
        )
    elif algorithm == "Random Forest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            criterion=params["criterion"],
            random_state=params["random_state"],
        )
    else:
        return LinearRegression(fit_intercept=params["fit_intercept"], n_jobs=params["n_jobs"])

def info(data_name, algorithm, algorithm_type, data, X, Y):
    if data_name not in ["Diabetes", "Salary", "Naive Bayes Classification", "Car Evaluation", "Heart Disease Classification", "Titanic"]:
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : {algorithm + " " + algorithm_type}')
        st.write("Shape of Dataset is: ", X.shape)
        st.write("Number of classes: ", len(np.unique(Y)))
        df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": data.target_names})
        st.write("Values and Name of Classes")
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")
    
    elif data_name == "Diabetes":
        st.write(f"## Regression {data_name} Dataset")
        st.write(f'Algorithm is : {algorithm + " " + algorithm_type}')
        st.write("Shape of Dataset is: ", X.shape)
    
    elif data_name == "Salary":
        st.write(f"## Regression {data_name} Dataset")
        st.write(f'Algorithm is : {algorithm + " " + algorithm_type}')
        st.write("Shape of Dataset is: ", X.shape)
    
    elif data_name == "Naive Bayes Classification":
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : {algorithm + " " + algorithm_type}')
        st.write("Shape of Dataset is: ", X.shape)
        st.write("Number of classes: ", len(np.unique(Y)))
        df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": ["Not Diabetic", "Diabetic"]})
        st.write("Values and Name of Classes")
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")
    
    elif data_name == "Heart Disease Classification":
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : {algorithm + " " + algorithm_type}')
        st.write("Shape of Dataset is: ", X.shape)
        st.write("Number of classes: ", len(np.unique(Y)))
        df = pd.DataFrame({
            "Target Value": list(np.unique(Y)),
            "Target Name": ["Less Chance Of Heart Attack", "High Chance Of Heart Attack"],
        })
        st.write("Values and Name of Classes")
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")
    
    elif data_name == "Titanic":
        st.write(f"## Classification {data_name} Dataset")
        st.write(f'Algorithm is : {algorithm + " " + algorithm_type}')
        st.write("Shape of Dataset is: ", X.shape)
        st.write("Number of classes: ", len(np.unique(Y)))
        df = pd.DataFrame({"Target Value": list(np.unique(Y)), "Target Name": ["Not Survived", "Survived"]})
        st.write("Values and Name of Classes")
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")
    
    else:
        st.write(f"## Classification {data_name} Dataset")
        st.write(f"Algorithm is : {algorithm}")
        st.write("Shape of Dataset is: ", X.shape)
        st.write("Number of classes: ", len(np.unique(Y)))
        df = pd.DataFrame({
            "Target Value": list(np.unique(Y)),
            "Target Name": ["Unacceptable", "Acceptable", "Good Condition", "Very Good Condition"],
        })
        st.write("Values and Name of Classes")
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.write("\n")

def choice_classifier(data, data_name, X, Y, fig, ax):
    if data_name == "Diabetes":
        ax.scatter(X[:, 0], X[:, 1], c=Y, cmap="viridis", alpha=0.8)
        ax.set_title("Scatter Classification Plot of Dataset")
        plt.colorbar(ax.scatter(X[:, 0], X[:, 1], c=Y, cmap="viridis", alpha=0.8), ax=ax)
    
    elif data_name == "Digits":
        colors = ["purple", "green", "yellow", "red", "black", "cyan", "pink", "magenta", "grey", "teal"]
        sns.scatterplot(x=X[:, 0], y=X[:, 1], data=data, c=Y, cmap='viridis', alpha=0.4, ax=ax)
        ax.legend(data.target_names, shadow=True)
        ax.set_title("Scatter Classification Plot of Dataset With Target Classes")
    
    elif data_name == "Salary":
        sns.scatterplot(x=data["YearsExperience"], y=data["Salary"], data=data, ax=ax)
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.set_title("Scatter Classification Plot of Dataset")
    
    elif data_name == "Naive Bayes Classification":
        colors = ["purple", "green"]
        sns.scatterplot(x=data["glucose"], y=data["bloodpressure"], data=data, hue=Y, palette=sns.color_palette(colors), alpha=0.4, ax=ax)
        ax.legend(shadow=True)
        ax.set_xlabel("Glucose")
        ax.set_ylabel("Blood Pressure")
        ax.set_title("Scatter Classification Plot of Dataset With Target Classes")
    
    else:
        colors = ["purple", "green", "yellow", "red"]
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette=sns.color_palette(colors), alpha=0.4, ax=ax)
        ax.legend(shadow=True)
        ax.set_title("Scatter Classification Plot of Dataset With Target Classes")
    
    return fig

def choice_regressor(X, x_test, predict, data, data_name, Y, fig, ax):
    if data_name == "Diabetes":
        ax.scatter(X[:, 0], Y, c=Y, cmap="viridis", alpha=0.4)
        ax.plot(x_test, predict, color="red")
        ax.set_title("Scatter Regression Plot of Dataset")
        ax.legend(["Actual Values", "Best Line or General formula"])
        plt.colorbar(ax.scatter(X[:, 0], Y, c=Y, cmap="viridis", alpha=0.4), ax=ax)
    
    elif data_name == "Digits":
        colors = ["purple", "green", "yellow", "red", "black", "cyan", "pink", "magenta", "grey", "teal"]
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette=sns.color_palette(colors), cmap="viridis", alpha=0.4, ax=ax)
        ax.plot(x_test, predict, color="red")
        ax.legend(data.target_names, shadow=True)
        ax.set_title("Scatter Plot of Dataset With Target Classes")
    
    elif data_name == "Salary":
        sns.scatterplot(x=data["YearsExperience"], y=data["Salary"], data=data, ax=ax)
        ax.plot(x_test, predict, color="red")
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.legend(["Actual Values", "Best Line or General formula"])
        ax.set_title("Scatter Regression Plot of Dataset")
    
    else:
        scat = ax.scatter(X[:, 0], X[:, 1], cmap="viridis", c=Y, alpha=0.4)
        ax.plot(x_test, predict, color="red")
        ax.legend(["Actual Values", "Best Line or General formula"])
        plt.colorbar(scat, ax=ax)
        ax.set_title("Scatter Regression Plot of Dataset With Target Classes")
    
    return fig


# NEW FEATURE 4: Model Comparison Dashboard
def show_model_history():
    st.subheader("🏆 Model Performance History")
    
    history = tracker.get_history()
    if not history:
        st.info("No models have been trained yet. Train some models to see the comparison!")
        return
    
    df = pd.DataFrame(history)
    
    # Performance comparison chart
    if len(df) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        df_sorted = df.sort_values('test_accuracy', ascending=True)
        ax1.barh(range(len(df_sorted)), df_sorted['test_accuracy'])
        ax1.set_yticks(range(len(df_sorted)))
        ax1.set_yticklabels([f"{row['model']} ({row['dataset']})" for _, row in df_sorted.iterrows()])
        ax1.set_xlabel('Test Accuracy')
        ax1.set_title('Model Performance Comparison')
        
        # Training time comparison
        ax2.bar(range(len(df)), df['training_time'])
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels([f"{row['model']}" for _, row in df.iterrows()], rotation=45)
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Comparison')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Show detailed table
    st.write("**Detailed Results:**")
    st.dataframe(df)
    
    # Export results
    if st.button("📥 Export Results as CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def data_model_description(algorithm, algorithm_type, data_name, data, X, Y):
    # Show dataset statistics
    show_dataset_statistics(data, data_name, X, Y)
    
    # Calling function to print Dataset Information
    info(data_name, algorithm, algorithm_type, data, X, Y)
    
    # Parameter selection
    if (algorithm_type == "Regressor") and (algorithm == "Decision Tree" or algorithm == "Random Forest" or algorithm_type == "Linear Regression"):
        params = add_parameter_regressor(algorithm)
    else:
        params = add_parameter_classifier_general(algorithm)
    
    # Model selection
    if algorithm_type == "Regressor":
        algo_model = model_regressor(algorithm, params)
    else:
        algo_model = model_classifier(algorithm, params)
    
    # Data splitting
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
    
    # Training with timing
    start_time = time.time()
    algo_model.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    predict = algo_model.predict(x_test)
    
    # PCA for plotting
    X_plot = pca_plot(data_name, X)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if algorithm_type == "Regressor":
        fig = choice_regressor(X_plot, x_test, predict, data, data_name, Y, fig, ax)
    else:
        fig = choice_classifier(data, data_name, X_plot, Y, fig, ax)
    
    if data_name != "Salary" and data_name != "Naive Bayes Classification":
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
    
    if data_name == "Naive Bayes Classification" and algorithm_type == "Regressor":
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
    
    st.pyplot(fig)
    
    # Performance metrics
    if algorithm != "Linear Regression" and algorithm_type != "Regressor":
        train_acc = algo_model.score(x_train, y_train) * 100
        test_acc = accuracy_score(y_test, predict) * 100
        st.write("Training Accuracy is:", train_acc)
        st.write("Testing Accuracy is:", test_acc)
        st.write(f"Training Time: {training_time:.4f} seconds")
        
        # Track performance
        tracker.add_performance(algorithm, data_name, algorithm_type, train_acc, test_acc, training_time)
    else:
        mse = mean_squared_error(y_test, predict)
        mae = mean_absolute_error(y_test, predict)
        st.write("Mean Squared error is:", mse)
        st.write("Mean Absolute error is:", mae)
        st.write(f"Training Time: {training_time:.4f} seconds")
        
        # Track performance (using negative MSE as score for regression)
        tracker.add_performance(algorithm, data_name, algorithm_type, -mse, -mse, training_time)
    
    # Show detailed metrics
   

def pca_plot(data_name, X):
    pca = PCA(2)
    if data_name != "Salary":
        X = pca.fit_transform(X)
    return X

# NEW FEATURE 5: Model Auto-Suggest
def suggest_best_algorithm(data_name, X, Y):
    st.subheader("🤖 Algorithm Recommendation")
    
    n_samples, n_features = X.shape
    n_classes = len(np.unique(Y))
    
    recommendations = []
    
    if n_samples < 1000:
        if n_features > 20:
            recommendations.append("Random Forest - Good for high-dimensional data")
        else:
            recommendations.append("SVM - Effective for small datasets")
    else:
        recommendations.append("Logistic Regression - Fast for large datasets")
    
    if n_classes == 2:
        recommendations.append("Naive Bayes - Excellent for binary classification")
    
    if n_features < 10:
        recommendations.append("KNN - Works well with low-dimensional data")
    
    st.write("**Recommended algorithms for this dataset:**")
    for rec in recommendations:
        st.write(f"• {rec}")

def main():
    st.title("🚀 Enhanced HyperTuneML Platform")
    st.write("### ML Algorithms on Inbuilt and Kaggle Datasets with Advanced Features")
    
    # NEW FEATURE: Tab navigation
    tab1, tab2, tab3 = st.tabs(["🔬 Train Models", "📊 Model Comparison", "🎯 Algorithm Recommendations"])
    
    with tab1:
        # Dataset selection
        data_name = st.sidebar.selectbox(
            "Select Dataset",
            ("Iris", "Breast Cancer", "Wine", "Diabetes", "Digits", "Salary",
             "Naive Bayes Classification", "Car Evaluation", "Heart Disease Classification", "Titanic"),
        )
        
        # Algorithm selection
        algorithm = st.sidebar.selectbox(
            "Select Supervised Learning Algorithm",
            ("KNN", "SVM", "Decision Tree", "Naive Bayes", "Random Forest", "Linear Regression", "Logistic Regression"),
        )
        
        # Algorithm type selection
        if algorithm not in ["Linear Regression", "Logistic Regression", "Naive Bayes"]:
            algorithm_type = st.sidebar.selectbox("Select Algorithm Type", ("Classifier", "Regressor"))
        else:
            st.sidebar.write(f"In {algorithm} Classifier and Regressor don't exist separately")
            if algorithm == "Linear Regression":
                algorithm_type = "Regressor"
                st.sidebar.write("{} only does Regression".format(algorithm))
            else:
                algorithm_type = "Classifier"
                st.sidebar.write(f"{algorithm} only does Classification")
        
        # Load dataset and run model
        data = load_dataset(data_name)
        X, Y = Input_output(data, data_name)
        
        data_model_description(algorithm, algorithm_type, data_name, data, X, Y)
    
    with tab2:
        show_model_history()
    
    with tab3:
        # Load dataset for recommendations
        data_name_rec = st.selectbox(
            "Select Dataset for Recommendation",
            ("Iris", "Breast Cancer", "Wine", "Diabetes", "Digits", "Salary",
             "Naive Bayes Classification", "Car Evaluation", "Heart Disease Classification", "Titanic"),
        )
        
        # Load the selected dataset
        data_rec = load_dataset(data_name_rec)
        X_rec, Y_rec = Input_output(data_rec, data_name_rec)
        
        # Show dataset info for recommendations
        st.write(f"### Dataset: {data_name_rec}")
        col1, col2, col3, col4 = st.columns(4)
        
        # Determine task type first
        if len(np.unique(Y_rec)) < 20:  # Classification task
            task_type = "Classification"
            n_classes = len(np.unique(Y_rec))
        else:  # Regression task
            task_type = "Regression"
            n_classes = len(np.unique(Y_rec))
        
        with col1:
            st.metric("Samples", X_rec.shape[0])
        with col2:
            st.metric("Features", X_rec.shape[1])
        with col3:
            st.metric("Classes", n_classes)
        with col4:
            # Calculate class balance for classification
            if task_type == "Classification" and n_classes < 10:
                class_counts = np.bincount(Y_rec.astype(int))
                balance_ratio = class_counts.min() / class_counts.max()
                st.metric("Class Balance", f"{balance_ratio:.2f}")
            else:
                st.metric("Class Balance", "N/A")
        
        # Generate recommendations
        suggest_best_algorithm(data_name_rec, X_rec, Y_rec)
        
        # Additional recommendations based on dataset characteristics
        st.subheader("📋 Detailed Analysis & Recommendations")
        
        n_samples, n_features = X_rec.shape
        n_classes = len(np.unique(Y_rec))
        
        # Dataset size analysis
        if n_samples < 500:
            st.warning("⚠️ Small dataset detected. Consider using cross-validation for better evaluation.")
        elif n_samples > 10000:
            st.info("💡 Large dataset detected. Linear models and ensemble methods work well.")
        
        # Feature analysis
        if n_features > n_samples:
            st.warning("⚠️ High-dimensional data (more features than samples). Consider feature selection or regularization.")
        
        # Class balance analysis (for classification)
        if task_type == "Classification" and n_classes < 10:
            class_counts = np.bincount(Y_rec.astype(int))
            balance_ratio = class_counts.min() / class_counts.max()
            
            if balance_ratio < 0.5:
                st.warning("⚠️ Imbalanced dataset detected. Consider using balanced algorithms or resampling techniques.")
            else:
                st.success("✅ Well-balanced dataset.")
        
        # Algorithm-specific recommendations
        st.subheader("🎯 Algorithm-Specific Recommendations")
        
        recommendations = {
            "KNN": {
                "best_for": "Small to medium datasets with clear cluster patterns",
                "avoid_if": "High-dimensional data or large datasets",
                "tip": "Use cross-validation to find optimal K value"
            },
            "SVM": {
                "best_for": "High-dimensional data and complex decision boundaries",
                "avoid_if": "Very large datasets (>10k samples)",
                "tip": "RBF kernel for non-linear problems, linear for high-dimensional data"
            },
            "Decision Tree": {
                "best_for": "Interpretable models and mixed data types",
                "avoid_if": "High risk of overfitting with deep trees",
                "tip": "Control max_depth to prevent overfitting"
            },
            "Random Forest": {
                "best_for": "Most datasets, especially with mixed features",
                "avoid_if": "Need highly interpretable models",
                "tip": "Generally robust, good default choice"
            },
            "Naive Bayes": {
                "best_for": "Text classification and when features are independent",
                "avoid_if": "Strong feature correlations exist",
                "tip": "Fast and works well with small datasets"
            },
            "Logistic Regression": {
                "best_for": "Binary classification and interpretable results",
                "avoid_if": "Complex non-linear relationships",
                "tip": "Add regularization for high-dimensional data"
            },
            "Linear Regression": {
                "best_for": "Simple linear relationships in regression",
                "avoid_if": "Non-linear relationships or many features",
                "tip": "Check for linear assumptions before using"
            }
        }
        
        for algo, info in recommendations.items():
            with st.expander(f"📊 {algo}"):
                st.write(f"**Best for:** {info['best_for']}")
                st.write(f"**Avoid if:** {info['avoid_if']}")
                st.write(f"**Tip:** {info['tip']}")
        
        # Quick start recommendation
        st.subheader("🚀 Quick Start Recommendation")
        
        if task_type == "Classification":
            if n_samples < 1000:
                if n_features < 20:
                    recommended_algo = "SVM with RBF kernel"
                else:
                    recommended_algo = "Random Forest"
            else:
                recommended_algo = "Logistic Regression"
        else:  # Regression
            if n_samples < 1000:
                recommended_algo = "Random Forest Regressor"
            else:
                recommended_algo = "Linear Regression"
        
        st.success(f"🎯 **Recommended starting algorithm:** {recommended_algo}")
        
        # Performance expectations
        st.subheader("📈 Expected Performance Guidelines")
        
        perf_guidelines = {
            "Excellent": "> 95% accuracy (classification) or R² > 0.9 (regression)",
            "Good": "85-95% accuracy (classification) or R² 0.7-0.9 (regression)", 
            "Fair": "70-85% accuracy (classification) or R² 0.5-0.7 (regression)",
            "Poor": "< 70% accuracy (classification) or R² < 0.5 (regression)"
        }
        
        for level, criteria in perf_guidelines.items():
            st.write(f"**{level}:** {criteria}")

if __name__ == "__main__":
    main()