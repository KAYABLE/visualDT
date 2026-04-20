import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import graphviz

# Load and Train
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=3) # Shorter depth for better "clinician" readability
model.fit(X_train, y_train)

st.set_page_config(layout="wide")
st.title("Diagnostic Decision Support System")

# --- SIDEBAR: PATIENT DATA ---
st.sidebar.header("Patient Vital Inputs")
user_input_vals = []
# Using relevant features for the slider ranges
for i in range(len(data.feature_names)):
    val = st.sidebar.number_input(f"{data.feature_names[i]}", value=float(X_test[0, i]))
    user_input_vals.append(val)

user_input = np.array(user_input_vals).reshape(1, -1)

# --- LAYOUT: PREDICTION & EXPLANATION ---
col1, col2 = st.columns(2)

with col1:
    prediction = model.predict(user_input)[0]
    label = "Benign" if prediction == 1 else "Malignant"
    st.metric("Diagnosis Prediction", label)
    
    # Clinician-Facing Report
    st.subheader("📋 Clinician Explanation Report")
    node_indicator = model.decision_path(user_input)
    leaf_id = model.apply(user_input)[0]
    
    st.write(f"**Case Summary:** The model classifies this sample as **{label}**.")
    st.write("**Key Decision Path:**")
    
    # Extract the path specific to this patient
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    for node_id in node_index:
        if feature[node_id] != -2: # Not a leaf
            name = data.feature_names[feature[node_id]]
            val = user_input[0, feature[node_id]]
            symbol = "<=" if val <= threshold[node_id] else ">"
            st.write(f"- {name} ({val:.2f}) {symbol} {threshold[node_id]:.2f}")

with col2:
    st.subheader("📊 Global Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    fig, ax = plt.subplots()
    ax.barh(data.feature_names[indices], importances[indices], color='skyblue')
    st.pyplot(fig)

# --- VISUAL DECISION TREE ---
st.divider()
st.subheader("🌲 Visual Decision Tree")
dot_data = export_graphviz(model, out_file=None, 
                           feature_names=data.feature_names, 
                           class_names=data.target_names, 
                           filled=True, rounded=True)
st.graphviz_chart(dot_data)

# --- RAW RULES ---
with st.expander("View Logic Rule Set (IF-THEN)"):
    st.text(export_text(model, feature_names=list(data.feature_names)))