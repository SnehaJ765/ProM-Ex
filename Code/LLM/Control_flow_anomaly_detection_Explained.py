import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
from collections import Counter
from itertools import islice
import numpy as np
import shap
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def clear_Page():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload a CSV file, and I will detect and explain their anomalies."}]


def main():
    st.set_page_config(
        page_title="ProM-Ex",
        page_icon="🍀"
    )

    with st.sidebar:
        st.title("Menu:")
        uploaded_file = st.file_uploader("Upload your CSV Files and Click on the Submit & Process Button", type="csv")
        submit_button = st.button("Submit & Process")

    st.title("ProM-Ex 🍀")
    st.write("Welcome! ")

    st.sidebar.button('Clear Page', on_click=clear_Page)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload a CSV file, and I will detect and explain their anomalies."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        case_id_col = st.selectbox("Select Case ID Column", data.columns)
        activity_col = st.selectbox("Select Activity Column", data.columns)

        data[case_id_col] = data[case_id_col].astype(str)
        data[activity_col] = data[activity_col].astype(str)

        activity_sequences = data.groupby(case_id_col)[activity_col].apply(list)

        def generate_ngrams(activities, n):
            return list(zip(*(islice(activities, i, None) for i in range(n))))

        n = 2
        activity_ngrams = activity_sequences.apply(lambda x: generate_ngrams(x, n))

        flattened_ngrams = [ngram for sublist in activity_ngrams for ngram in sublist]
        ngram_counts = Counter(flattened_ngrams)

        unique_ngrams = list(ngram_counts.keys())
        ngram_feature_matrix = []

        for case_id in activity_sequences.index:
            ngrams = generate_ngrams(activity_sequences[case_id], n)
            ngram_vector = [ngrams.count(ngram) for ngram in unique_ngrams]
            ngram_feature_matrix.append(ngram_vector)

        ngram_feature_matrix = np.array(ngram_feature_matrix)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(ngram_feature_matrix)

        iso_model = IsolationForest(contamination=0.01, random_state=42)
        iso_model.fit(X_scaled)

        def detect_anomalies(model, features):
            predictions = model.predict(features)
            anomalies = activity_sequences.iloc[predictions == -1]
            return predictions, anomalies

        iso_predictions, iso_anomalies = detect_anomalies(iso_model, X_scaled)

        true_labels = [1] * len(activity_sequences)
        iso_accuracy = accuracy_score(true_labels, iso_predictions)
        iso_f1 = f1_score(true_labels, iso_predictions, pos_label=1)
        iso_errors = len(iso_anomalies)

        st.write("Detected Anomalies:")
        st.dataframe(iso_anomalies)

        st.subheader("SHAP Analysis")

        if X_scaled.shape[1] > 0 and len(unique_ngrams) > 0:
            explainer = shap.TreeExplainer(iso_model)
            shap_values = explainer.shap_values(X_scaled)

            st.write("Feature Importance Plot")
            shap.summary_plot(shap_values, X_scaled, feature_names=unique_ngrams, plot_type="bar")
            st.pyplot()

            shap_values_mean = np.abs(shap_values).mean(axis=0)
            top_ngram_index = np.argmax(shap_values_mean)
            top_ngram_name = str(unique_ngrams[top_ngram_index])  # Convert tuple to string

        st.subheader("LLM-based SHAP Explanation")
        shap_values_abs = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(unique_ngrams, shap_values_abs))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        explanation = generate_shap_explanation(top_features)
        st.write(explanation)


def generate_shap_explanation(top_features):
    """
    Generate a text-based explanation for SHAP feature importances using an LLM.

    Parameters:
    top_features (list): A list of tuples containing the top features and their importance scores.

    Returns:
    str: A text explanation of feature importances.
    """

    feature_names = ', '.join([str(f[0]) for f in top_features])
    feature_importances = ', '.join([f"{f[0]}: {f[1]:.2f}" for f in top_features])

    prompt_template = """
In the context of process mining, analyze the types of control-flow anomalies detected using the following n-gram features and their associated importance scores in anomaly detection:

Feature Importance (Top n-gram Sequences and Scores):
{feature_importances}

For each feature, determine the likely type of control-flow anomaly (e.g., unexpected event order, missing events, redundant events, or looping patterns). Additionally, provide detailed insights on the following:

Significance of Features: Explain why these specific features are critical in highlighting control-flow anomalies and how they signal potential deviations from the expected process flow.
Impact on Process Flow: Describe the effects of these anomalies on the overall process, including potential delays, errors, inefficiencies, or other disruptions they may cause in the workflow.
Use a structured response with clear explanations, relevant examples, and actionable recommendations to support process improvement.
"""



    context = "The context information about how these features relate to the anomalies detected in the process mining."
    prompt = prompt_template.format(
        feature_importances=feature_importances,
        context=context
    )

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, google_api_key=os.getenv("GOOGLE_API_KEY"))
    response = model.predict(prompt)

    return response


if __name__ == "__main__":
    main()