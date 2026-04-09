import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

# Page Configuration
st.set_page_config(page_title="Decision Tree Classifier", layout="wide")

st.title("🌳 Decision Tree Classification Dashboard")
st.markdown("""
Interactive visualization of the **Decision Tree** algorithm. Unlike Naive Bayes, Decision Trees create 
**orthogonal (rectangular) decision boundaries**. Adjust the depth to see how it captures data patterns!
""")

# --- Sidebar: Configuration ---
st.sidebar.header("Model Hyperparameters")

# Hyperparameters
criterion = st.sidebar.selectbox("Criterion", ("entropy", "gini"))
max_depth = st.sidebar.slider("Max Depth", 1, 10, 5)
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 25) / 100
random_state = st.sidebar.number_input("Random State", value=0)

# --- Data Processing ---
@st.cache_data
def load_data():
    # Attempting to load the local file
    try:
        return pd.read_csv('Social_Network_Ads.csv')
    except:
        st.error("CSV file not found! Please ensure 'Social_Network_Ads.csv' is in the directory.")
        return None

dataset = load_data()

if dataset is not None:
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Feature Scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Training
    classifier = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)
    classifier.fit(X_train_scaled, y_train)

    # --- Main Panel ---
    tab1, tab2 = st.tabs(["📈 Classification & Metrics", "🌿 Tree Structure"])

    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🎯 Test Prediction")
            age = st.slider("Select Age", 18, 60, 30)
            salary = st.slider("Select Salary", 15000, 150000, 87000)
            
            prediction = classifier.predict(sc.transform([[age, salary]]))
            result = "Purchased" if prediction[0] == 1 else "Not Purchased"
            st.info(f"Model Prediction: **{result}**")

            st.divider()
            
            st.subheader("📊 Metrics")
            y_pred = classifier.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.2%}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            ConfusionMatrixDisplay(cm).plot(ax=ax_cm, cmap='Greens', colorbar=False)
            st.pyplot(fig_cm)

        with col2:
            st.subheader("🗺️ Decision Boundary")
            mode = st.radio("Visualize:", ("Training Set", "Test Set"), horizontal=True)
            
            def plot_decision_boundary(X_data, y_data, title):
                X_set, y_set = sc.inverse_transform(X_data), y_data
                # Grid setup
                X1, X2 = np.meshgrid(
                    np.arange(start=X_set[:, 0].min() - 5, stop=X_set[:, 0].max() + 5, step=1),
                    np.arange(start=X_set[:, 1].min() - 500, stop=X_set[:, 1].max() + 500, step=500)
                )
                
                # Boundary prediction
                Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
                
                fig, ax = plt.subplots()
                ax.contourf(X1, X2, Z, alpha=0.6, cmap=ListedColormap(['#FA8072', '#1E90FF']))
                
                for i, j in enumerate(np.unique(y_set)):
                    ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                               c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j, edgecolors='white')
                
                ax.set_title(title)
                ax.set_xlabel('Age')
                ax.set_ylabel('Estimated Salary')
                ax.legend()
                return fig

            if mode == "Training Set":
                st.pyplot(plot_decision_boundary(X_train_scaled, y_train, "DT (Training Set)"))
            else:
                st.pyplot(plot_decision_boundary(X_test_scaled, y_test, "DT (Test Set)"))

    with tab2:
        st.subheader("Structure of the Decision Tree")
        st.write("This shows how the algorithm is splitting the features internally.")
        fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
        plot_tree(classifier, 
                  feature_names=['Age', 'Salary'], 
                  class_names=['No', 'Yes'], 
                  filled=True, 
                  rounded=True, 
                  fontsize=12, 
                  ax=ax_tree)
        st.pyplot(fig_tree)

    # Data Preview
    with st.expander("🔍 Dataset Overview"):
        st.dataframe(dataset.head(10))
