# src/models/predict_model.py

import os
import sys
import pandas as pd
import numpy as np
import pickle
import joblib
import logging
import json
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project directory to system path to enable imports from other project modules
project_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(project_dir))

# Define paths to model artifacts
MODEL_DIR = project_dir / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "loan_default_prediction_model.pkl"
DEFAULT_THRESHOLD_PATH = MODEL_DIR / "optimal_threshold.pkl"
DEFAULT_FEATURES_PATH = MODEL_DIR / "feature_names.pkl"
DEFAULT_METADATA_PATH = MODEL_DIR / "model_metadata.json"


def load_model(model_path: Optional[Union[str, Path]] = None) -> Any:
    """
    Load the trained prediction model.
    
    Parameters
    ----------
    model_path : str or Path, optional
        Path to the model file. If None, uses the default path.
    
    Returns
    -------
    model
        The trained model object
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_threshold(threshold_path: Optional[Union[str, Path]] = None) -> float:
    """
    Load the optimal classification threshold.
    
    Parameters
    ----------
    threshold_path : str or Path, optional
        Path to the threshold file. If None, uses the default path.
    
    Returns
    -------
    float
        The optimal classification threshold
    """
    if threshold_path is None:
        threshold_path = DEFAULT_THRESHOLD_PATH
    
    threshold_path = Path(threshold_path)
    
    if not threshold_path.exists():
        logger.warning(f"Threshold file not found at: {threshold_path}. Using default threshold of 0.5.")
        return 0.5
    
    try:
        with open(threshold_path, 'rb') as f:
            threshold = pickle.load(f)
        return threshold
    except Exception as e:
        logger.error(f"Error loading threshold: {e}. Using default threshold of 0.5.")
        return 0.5


def load_feature_names(features_path: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Load the list of feature names used by the model.
    
    Parameters
    ----------
    features_path : str or Path, optional
        Path to the features file. If None, uses the default path.
    
    Returns
    -------
    list of str
        The list of feature names
    """
    if features_path is None:
        features_path = DEFAULT_FEATURES_PATH
    
    features_path = Path(features_path)
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at: {features_path}")
    
    try:
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        return feature_names
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")
        raise


def load_model_metadata(metadata_path: Optional[Union[str, Path]] = None) -> Dict:
    """
    Load model metadata including performance metrics and other information.
    
    Parameters
    ----------
    metadata_path : str or Path, optional
        Path to the metadata file. If None, uses the default path.
    
    Returns
    -------
    dict
        The model metadata
    """
    if metadata_path is None:
        metadata_path = DEFAULT_METADATA_PATH
    
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found at: {metadata_path}. Returning empty metadata.")
        return {}
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return {}


def validate_input_data(data: pd.DataFrame, feature_names: List[str]) -> Tuple[bool, str]:
    """
    Validate that the input data contains all required features and has valid values.
    
    Parameters
    ----------
    data : pandas DataFrame
        The input data to validate
    feature_names : list of str
        The required feature names
    
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    # Check if all required features are present
    missing_features = set(feature_names) - set(data.columns)
    if missing_features:
        return False, f"Missing required features: {missing_features}"
    
    # Check for null values
    null_columns = data[feature_names].columns[data[feature_names].isnull().any()].tolist()
    if null_columns:
        return False, f"Null values found in columns: {null_columns}"
    
    # Check for non-numeric values in numeric columns
    non_numeric_columns = []
    for col in feature_names:
        if not pd.api.types.is_numeric_dtype(data[col]):
            try:
                # Try to convert to numeric
                data[col] = pd.to_numeric(data[col])
            except:
                non_numeric_columns.append(col)
    
    if non_numeric_columns:
        return False, f"Non-numeric values found in columns: {non_numeric_columns}"
    
    return True, ""


def preprocess_data(data: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    Prepare the input data for prediction by ensuring correct order and types.
    
    Parameters
    ----------
    data : pandas DataFrame
        The input data to preprocess
    feature_names : list of str
        The required feature names in the correct order
    
    Returns
    -------
    pandas DataFrame
        The preprocessed data ready for prediction
    """
    # Select only the required features in the correct order
    preprocessed_data = data[feature_names].copy()
    
    # Ensure all data is numeric
    for col in preprocessed_data.columns:
        preprocessed_data[col] = pd.to_numeric(preprocessed_data[col], errors='coerce')
    
    # Fill any remaining NaN values with 0 (this is a conservative approach)
    # In a real scenario, you might want to impute values based on training data statistics
    if preprocessed_data.isnull().any().any():
        logger.warning("NaN values found after preprocessing. Filling with zeros.")
        preprocessed_data.fillna(0, inplace=True)
    
    return preprocessed_data


def predict_loan_default(
    data: Union[pd.DataFrame, Dict, List[Dict]],
    model_path: Optional[Union[str, Path]] = None,
    threshold_path: Optional[Union[str, Path]] = None,
    features_path: Optional[Union[str, Path]] = None,
    custom_threshold: Optional[float] = None
) -> Dict:
    """
    Predict loan default probability and risk segment for new applications.
    
    Parameters
    ----------
    data : pandas DataFrame or dict or list of dicts
        Data containing the features needed for prediction
    model_path : str or Path, optional
        Path to the model file. If None, uses the default path.
    threshold_path : str or Path, optional
        Path to the threshold file. If None, uses the default path.
    features_path : str or Path, optional
        Path to the features file. If None, uses the default path.
    custom_threshold : float, optional
        Custom threshold to use for classification. If provided, overrides the loaded threshold.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'probabilities': Array of predicted default probabilities
        - 'predictions': Array of binary predictions (0=No Default, 1=Default)
        - 'risk_segments': Array of risk segments ('Very Low', 'Low', 'Medium', 'High', 'Very High')
        - 'feature_importance': List of (feature, importance) tuples for the most important features
          (only for single predictions)
    """
    # Convert input to DataFrame if it's not already
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        data = pd.DataFrame(data)
    elif not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame, a dictionary, or a list of dictionaries")
    
    # Load model artifacts
    model = load_model(model_path)
    threshold = custom_threshold if custom_threshold is not None else load_threshold(threshold_path)
    feature_names = load_feature_names(features_path)
    
    # Validate input data
    valid, error_message = validate_input_data(data, feature_names)
    if not valid:
        raise ValueError(f"Invalid input data: {error_message}")
    
    # Preprocess data
    preprocessed_data = preprocess_data(data, feature_names)
    
    logger.info(f"Making predictions for {len(preprocessed_data)} applications")
    
    # Make predictions
    try:
        probabilities = model.predict_proba(preprocessed_data)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Create risk segments
        risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        risk_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        risk_segments = pd.cut(probabilities, bins=risk_bins, labels=risk_labels).astype(str)
        
        # Get feature importance for single predictions
        feature_importance = []
        if len(preprocessed_data) == 1 and hasattr(model, 'feature_importances_'):
            # Get feature importance from the model
            importances = model.feature_importances_
            # Create a list of (feature, importance) tuples
            feature_importance = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
        
        # Return results
        results = {
            'probabilities': probabilities,
            'predictions': predictions,
            'risk_segments': risk_segments
        }
        
        if feature_importance:
            results['feature_importance'] = feature_importance
        
        return results
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


def get_risk_description(risk_segment: str) -> str:
    """
    Get a standardized description of what a risk segment means.
    
    Parameters
    ----------
    risk_segment : str
        The risk segment ('Very Low', 'Low', 'Medium', 'High', 'Very High')
    
    Returns
    -------
    str
        Description of the risk segment and recommended actions
    """
    descriptions = {
        'Very Low': (
            "Very low risk of default. This application shows strong indicators of repayment ability. "
            "Recommended for approval with standard or preferential terms."
        ),
        'Low': (
            "Low risk of default. This application has positive indicators of repayment ability. "
            "Recommended for approval with standard terms."
        ),
        'Medium': (
            "Medium risk of default. This application shows moderate risk factors. "
            "Consider additional verification, a co-signer, or adjusted terms."
        ),
        'High': (
            "High risk of default. This application has significant risk factors. "
            "Consider stronger collateral requirements, higher interest rate, or shorter term."
        ),
        'Very High': (
            "Very high risk of default. This application has multiple strong risk indicators. "
            "Approval not recommended without substantial additional guarantees or collateral."
        )
    }
    
    return descriptions.get(risk_segment, "Unknown risk segment")


def generate_loan_report(
    application_data: Dict,
    prediction_results: Dict,
    include_recommendations: bool = True
) -> str:
    """
    Generate a human-readable report for a loan application prediction.
    
    Parameters
    ----------
    application_data : dict
        The loan application data
    prediction_results : dict
        The prediction results from predict_loan_default
    include_recommendations : bool, default=True
        Whether to include recommendations in the report
    
    Returns
    -------
    str
        A formatted report string
    """
    # Extract prediction results for the first (and only) application
    probability = prediction_results['probabilities'][0]
    prediction = prediction_results['predictions'][0]
    risk_segment = prediction_results['risk_segments'][0]
    feature_importance = prediction_results.get('feature_importance', [])
    
    # Create the report
    report = []
    report.append("=== LOAN DEFAULT RISK ASSESSMENT REPORT ===")
    report.append("")
    
    # Application summary
    report.append("APPLICATION SUMMARY:")
    report.append("-" * 30)
    for key, value in application_data.items():
        report.append(f"{key.replace('_', ' ').title()}: {value}")
    report.append("")
    
    # Prediction results
    report.append("RISK ASSESSMENT:")
    report.append("-" * 30)
    report.append(f"Default Probability: {probability:.2%}")
    report.append(f"Prediction: {'Default' if prediction == 1 else 'No Default'}")
    report.append(f"Risk Segment: {risk_segment}")
    report.append("")
    
    # Risk description
    if include_recommendations:
        report.append("RECOMMENDATION:")
        report.append("-" * 30)
        report.append(get_risk_description(risk_segment))
        report.append("")
    
    # Key factors (if available)
    if feature_importance:
        report.append("KEY RISK FACTORS:")
        report.append("-" * 30)
        for feature, importance in feature_importance[:5]:  # Top 5 features
            report.append(f"- {feature.replace('_', ' ').title()}: {importance:.4f}")
        report.append("")
    
    return "\n".join(report)


def format_predictions_as_dataframe(
    data: pd.DataFrame,
    prediction_results: Dict
) -> pd.DataFrame:
    """
    Format prediction results as a pandas DataFrame with the original data.
    
    Parameters
    ----------
    data : pandas DataFrame
        The original input data
    prediction_results : dict
        The prediction results from predict_loan_default
    
    Returns
    -------
    pandas DataFrame
        A DataFrame with the original data and prediction results
    """
    # Create a copy of the original data
    result_df = data.copy()
    
    # Add prediction columns
    result_df['default_probability'] = prediction_results['probabilities']
    result_df['default_prediction'] = prediction_results['predictions']
    result_df['risk_segment'] = prediction_results['risk_segments']
    
    # Map binary predictions to readable values
    result_df['default_prediction'] = result_df['default_prediction'].map({
        0: 'No Default',
        1: 'Default'
    })
    
    return result_df


def main():
    """
    Command-line interface for the prediction module.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict loan default risk.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('--output', '-o', type=str, help='Path to save output CSV file')
    parser.add_argument('--threshold', '-t', type=float, help='Custom classification threshold')
    parser.add_argument('--model', '-m', type=str, help='Path to model file')
    parser.add_argument('--features', '-f', type=str, help='Path to feature names file')
    
    args = parser.parse_args()
    
    try:
        # Load input data
        logger.info(f"Loading input data from {args.input_file}")
        input_data = pd.read_csv(args.input_file)
        
        # Make predictions
        results = predict_loan_default(
            data=input_data,
            model_path=args.model,
            threshold_path=None,  # Use default
            features_path=args.features,
            custom_threshold=args.threshold
        )
        
        # Format results
        output_df = format_predictions_as_dataframe(input_data, results)
        
        # Save or display results
        if args.output:
            output_df.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")
        else:
            # Display summary
            total = len(output_df)
            defaults = (output_df['default_prediction'] == 'Default').sum()
            logger.info(f"Prediction summary: {defaults} defaults out of {total} applications ({defaults/total:.1%})")
            
            # Display risk segment distribution
            risk_counts = output_df['risk_segment'].value_counts().sort_index()
            for segment, count in risk_counts.items():
                logger.info(f"  {segment}: {count} applications ({count/total:.1%})")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())