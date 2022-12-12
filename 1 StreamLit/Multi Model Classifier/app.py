import streamlit as st
from model import CommonModel

st.title("Machine Learning Web App 01")

st.write("""
# Explore different classifier
## Which one is the best?
""")

# Add widgets to the app
ds_name = st.sidebar.selectbox("Select Dataset", ("Iris", "BreastCancer", "WineQuality"))
clf_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "RF"))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(clf_name)
cm = CommonModel(clf_name, ds_name, params)
cm.process()

acc = cm.score()
st.write("""##### Classifier  ->  """, clf_name)
st.write("""##### Accuracy    ->  """, acc)