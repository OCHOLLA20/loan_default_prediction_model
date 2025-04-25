import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import shap
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from streamlit_shap import st_shap

# Set page configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("Loan Default Risk Assessment Tool")
st.markdown("""
This application helps predict the probability of loan default based on customer and loan characteristics.
It uses a machine learning model trained on historical loan data from a Kenyan lending institution.
""")

# Define paths
MODEL_DIR = Path("../models")
model_path = MODEL_DIR / "loan_default_prediction_model.pkl"
threshold_path = MODEL_DIR / "optimal_threshold.pkl"
features_path = MODEL_DIR / "feature_names.pkl"
metadata_path = MODEL_DIR / "model_metadata.json"

# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    """Load model and related artifacts"""
    model = joblib.load(model_path)
    
    with open(threshold_path, 'rb') as f:
        threshold = pickle.load(f)
        
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, threshold, feature_names

# Load model metadata
@st.cache_data
def load_model_metadata():
    """Load model metadata"""
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

# Make predictions using the loaded model
def predict_loan_default(data, model, threshold, feature_names):
    """
    Predict loan default probability and binary outcome.
    
    Parameters:
    -----------
    data : pandas DataFrame
        Data containing the features needed for prediction
    model : trained model
        The trained model
    threshold : float
        Classification threshold
    feature_names : list
        List of feature names
    
    Returns:
    --------
    dict
        Dictionary containing probabilities and binary predictions
    """
    # Ensure data has all required features
    missing_features = set(feature_names) - set(data.columns)
    if missing_features:
        st.error(f"Missing features in input data: {missing_features}")
        st.stop()
    
    # Select and order features correctly
    X = data[feature_names]
    
    # Make predictions
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # Create risk segments
    risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    risk_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    risk_segments = pd.cut(probabilities, bins=risk_bins, labels=risk_labels).astype(str)
    
    # Return results
    return {
        'probabilities': probabilities,
        'predictions': predictions,
        'risk_segments': risk_segments
    }

# SHAP explanation function
@st.cache_data
def generate_shap_explanation(model, X, _feature_names):
    """Generate SHAP values for model explanation"""
    # Create explainer
    explainer = shap.Explainer(model)
    # Calculate SHAP values
    shap_values = explainer(X)
    return shap_values

# Function to generate example data
def generate_example_data(feature_names):
    """Generate example data with reasonable values for demonstration"""
    example_data = {
        # Add reasonable default values for each feature
        # These would be populated with domain-specific reasonable values
        # For example:
        'loan_amount': 50000,
        'interest_rate': 15.0,
        'loan_term': 12,
        'credit_score': 650,
        # ... other features
    }
    
    # Create a DataFrame with only the required features
    example_df = pd.DataFrame([example_data])
    
    # Fill missing features with reasonable defaults
    for feature in feature_names:
        if feature not in example_df.columns:
            # Assign reasonable default based on feature name
            if 'amount' in feature.lower():
                example_df[feature] = 50000
            elif 'rate' in feature.lower():
                example_df[feature] = 15.0
            elif 'term' in feature.lower():
                example_df[feature] = 12
            elif 'age' in feature.lower():
                example_df[feature] = 35
            elif 'score' in feature.lower():
                example_df[feature] = 650
            elif 'ratio' in feature.lower():
                example_df[feature] = 0.5
            elif 'count' in feature.lower():
                example_df[feature] = 5
            elif 'days' in feature.lower():
                example_df[feature] = 0
            elif 'flag' in feature.lower() or feature.startswith('has_'):
                example_df[feature] = 0
            else:
                example_df[feature] = 0  # Generic default
    
    return example_df

try:
    # Load model and artifacts
    model, threshold, feature_names = load_model_artifacts()
    metadata = load_model_metadata()
    
    # Sidebar with model information
    with st.sidebar:
        st.header("Model Information")
        st.markdown(f"**Model Type:** {metadata['model_type']}")
        st.markdown(f"**Training Date:** {metadata['training_date']}")
        st.markdown(f"**Optimal Threshold:** {metadata['optimal_threshold']:.3f}")
        
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
            'Value': [
                metadata['performance_metrics']['accuracy'],
                metadata['performance_metrics']['precision'],
                metadata['performance_metrics']['recall'],
                metadata['performance_metrics']['f1'],
                metadata['performance_metrics']['auc_roc']
            ]
        })
        st.dataframe(metrics_df.set_index('Metric'), use_container_width=True)
        
        st.subheader("Top Features")
        st.write(", ".join(metadata['top_features'][:5]))
        
        st.subheader("About")
        st.info("This tool uses machine learning to predict loan default risk. It helps lenders make informed decisions and manage their loan portfolio effectively.")

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Upload Data", "Single Prediction", "Documentation"])
    
    with tab1:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with loan applications to get predictions for multiple applications at once.")
        
        uploaded_file = st.file_uploader("Choose a CSV file with loan application data", type="csv")
        
        if uploaded_file is not None:
            # Read the uploaded file
            input_df = pd.read_csv(uploaded_file)
            
            # Show preview of uploaded data
            st.subheader("Data Preview")
            st.dataframe(input_df.head(), use_container_width=True)
            
            # Make predictions
            with st.spinner("Generating predictions..."):
                results = predict_loan_default(input_df, model, threshold, feature_names)
                
                # Add predictions to dataframe
                output_df = input_df.copy()
                output_df["Default Probability"] = results["probabilities"]
                output_df["Default Prediction"] = results["predictions"].map({0: "No Default", 1: "Default"})
                output_df["Risk Segment"] = results["risk_segments"]
                
                # Show results
                st.subheader("Prediction Results")
                st.dataframe(output_df, use_container_width=True)
                
                # Visualize results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk segment distribution
                    risk_counts = output_df["Risk Segment"].value_counts().reset_index()
                    risk_counts.columns = ["Risk Segment", "Count"]
                    
                    # Sort by risk level
                    risk_order = ["Very Low", "Low", "Medium", "High", "Very High"]
                    risk_counts["Risk Segment"] = pd.Categorical(
                        risk_counts["Risk Segment"], 
                        categories=risk_order, 
                        ordered=True
                    )
                    risk_counts = risk_counts.sort_values("Risk Segment")
                    
                    fig1 = px.bar(
                        risk_counts, 
                        x="Risk Segment", 
                        y="Count",
                        color="Risk Segment",
                        color_discrete_map={
                            "Very Low": "#2ecc71",
                            "Low": "#3498db",
                            "Medium": "#f1c40f",
                            "High": "#e67e22",
                            "Very High": "#e74c3c"
                        },
                        title="Distribution by Risk Segment"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Probability distribution
                    fig2 = px.histogram(
                        output_df, 
                        x="Default Probability", 
                        nbins=20, 
                        title="Distribution of Default Probabilities"
                    )
                    fig2.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold:.2f}")
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Download results
                csv = output_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="loan_default_predictions.csv",
                    mime="text/csv",
                )
                
                # SHAP explanation for a sample
                if st.checkbox("Show prediction explanations for a sample"):
                    sample_size = min(5, len(input_df))
                    sample_indices = st.multiselect(
                        "Select rows to explain (up to 5)",
                        options=list(range(len(input_df))),
                        default=list(range(min(3, len(input_df))))
                    )
                    
                    if sample_indices:
                        sample_df = input_df.iloc[sample_indices]
                        shap_values = generate_shap_explanation(model, sample_df[feature_names], feature_names)
                        
                        st.subheader("Feature Impact on Predictions")
                        st_shap(shap.plots.waterfall(shap_values[0]), height=300)
                        st_shap(shap.plots.beeswarm(shap_values), height=300)
    
    with tab2:
        st.header("Single Loan Assessment")
        st.markdown("Enter loan application details to get an individual risk assessment.")
        
        # Generate example data for form default values
        example_data = generate_example_data(feature_names)
        
        # Create form for manual input
        with st.form("loan_application_form"):
            st.subheader("Loan Application Details")
            
            # Organize features into categories for better UX
            # These categories would be customized based on actual features
            st.markdown("### Loan Information")
            col1, col2, col3 = st.columns(3)
            loan_features = {}
            
            for i, feature in enumerate(feature_names):
                # Place input fields in different columns for better layout
                with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
                    # Create appropriate input widget based on feature name
                    if 'amount' in feature.lower():
                        loan_features[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}", 
                            min_value=0.0, 
                            value=float(example_data[feature][0]),
                            step=1000.0,
                            format="%.2f"
                        )
                    elif 'rate' in feature.lower():
                        loan_features[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}", 
                            min_value=0.0, 
                            max_value=100.0,
                            value=float(example_data[feature][0]),
                            step=0.1,
                            format="%.2f"
                        )
                    elif feature.endswith('flag') or feature.startswith('has_'):
                        loan_features[feature] = st.selectbox(
                            f"{feature.replace('_', ' ').title()}", 
                            options=[0, 1],
                            index=int(example_data[feature][0])
                        )
                    else:
                        loan_features[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}", 
                            value=float(example_data[feature][0]),
                            step=0.01 if 'ratio' in feature.lower() else 1.0
                        )
            
            submitted = st.form_submit_button("Predict Default Risk")
        
        # Process form submission
        if submitted:
            # Create DataFrame from form input
            input_data = pd.DataFrame([loan_features])
            
            # Make prediction
            with st.spinner("Analyzing loan application..."):
                results = predict_loan_default(input_data, model, threshold, feature_names)
                
                # Display results
                st.subheader("Risk Assessment Results")
                
                # Create columns for different result aspects
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Default probability gauge
                    probability = results["probabilities"][0]
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Default Probability"},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.2], 'color': "green"},
                                {'range': [0.2, 0.4], 'color': "limegreen"},
                                {'range': [0.4, 0.6], 'color': "yellow"},
                                {'range': [0.6, 0.8], 'color': "orange"},
                                {'range': [0.8, 1], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': threshold
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk segment
                    risk_segment = results["risk_segments"][0]
                    risk_colors = {
                        "Very Low": "#2ecc71",
                        "Low": "#3498db",
                        "Medium": "#f1c40f",
                        "High": "#e67e22",
                        "Very High": "#e74c3c"
                    }
                    st.markdown(f"""
                        <div style="background-color: {risk_colors[risk_segment]}; padding: 20px; border-radius: 10px; text-align: center;">
                            <h1 style="color: white;">Risk Segment</h1>
                            <h2 style="color: white;">{risk_segment}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Decision
                    prediction = results["predictions"][0]
                    decision = "Likely to Default" if prediction == 1 else "Likely to Repay"
                    color = "#e74c3c" if prediction == 1 else "#2ecc71"
                    st.markdown(f"""
                        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                            <h1 style="color: white;">Decision</h1>
                            <h2 style="color: white;">{decision}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                # SHAP explanation
                st.subheader("What factors influenced this prediction?")
                with st.spinner("Generating explanation..."):
                    shap_values = generate_shap_explanation(model, input_data[feature_names], feature_names)
                    
                    # Waterfall plot for feature contributions
                    st_shap(shap.plots.waterfall(shap_values[0]), height=400)
                    
                    # Additional recommendations based on risk level
                    st.subheader("Recommendations")
                    if risk_segment in ["High", "Very High"]:
                        st.error("""
                            This application presents a significant default risk. Consider:
                            - Requesting additional collateral
                            - Offering a shorter loan term
                            - Increasing the interest rate to compensate for risk
                            - Implementing stricter monitoring procedures
                        """)
                    elif risk_segment == "Medium":
                        st.warning("""
                            This application presents a moderate default risk. Consider:
                            - Reviewing the applicant's payment history more carefully
                            - Requesting a co-signer
                            - Offering a slightly higher interest rate
                        """)
                    else:
                        st.success("""
                            This application presents a low default risk. Consider:
                            - Proceeding with standard loan terms
                            - Offering preferred interest rates
                            - Using as an opportunity for cross-selling other financial products
                        """)
    
    with tab3:
        st.header("Documentation")
        
        st.subheader("How to Use This Tool")
        st.markdown("""
        This application helps assess the risk of loan default for new loan applications. There are two main ways to use it:
        
        1. **Batch Processing**: Upload a CSV file with multiple loan applications to get predictions for all of them at once.
           - Your CSV should include all the required features used by the model
           - Results can be downloaded as a CSV file with added prediction columns
        
        2. **Single Application**: Enter details for a single loan application to get a detailed risk assessment.
           - The form includes all required features
           - You'll get a visual representation of the risk level and factors influencing it
           - Recommendations are provided based on the risk assessment
        """)
        
        st.subheader("Understanding the Results")
        st.markdown("""
        The model provides several outputs:
        
        - **Default Probability**: A score between 0 and 1 indicating the likelihood of default
        - **Default Prediction**: A binary outcome (Default/No Default) based on the optimal threshold
        - **Risk Segment**: A categorical risk level from Very Low to Very High
        
        Risk segments are defined as follows:
        
        | Risk Level | Default Probability | Typical Default Rate |
        |------------|---------------------|----------------------|
        | Very Low   | 0.0 - 0.2           | ~5%                  |
        | Low        | 0.2 - 0.4           | ~15%                 |
        | Medium     | 0.4 - 0.6           | ~35%                 |
        | High       | 0.6 - 0.8           | ~60%                 |
        | Very High  | 0.8 - 1.0           | ~85%                 |
        """)
        
        st.subheader("Model Information")
        st.markdown(f"""
        This tool uses a {metadata['model_type']} model trained on historical loan data from a Kenyan lending institution.
        
        **Key Performance Metrics**:
        - Accuracy: {metadata['performance_metrics']['accuracy']:.2f}
        - Precision: {metadata['performance_metrics']['precision']:.2f}
        - Recall: {metadata['performance_metrics']['recall']:.2f}
        - F1 Score: {metadata['performance_metrics']['f1']:.2f}
        - AUC-ROC: {metadata['performance_metrics']['auc_roc']:.2f}
        
        **Important Features**:
        The model considers many factors, with the most influential being:
        {', '.join(metadata['top_features'][:5])}
        """)
        
        st.subheader("Interpretation Guide")
        st.markdown("""
        The SHAP (SHapley Additive exPlanations) charts show how each feature affects the prediction:
        
        - **Waterfall Plot**: Shows how each feature pushes the prediction up (red) or down (blue) from the base value
        - **Beeswarm Plot**: Shows the distribution of feature effects across multiple applications
        
        These visualizations help understand why a particular application received its risk score and can guide mitigation strategies.
        """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.markdown("""
    ### Troubleshooting
    
    Please ensure:
    1. The model files are available in the correct location
    2. The input data contains all required features
    3. The feature values are within expected ranges
    """)