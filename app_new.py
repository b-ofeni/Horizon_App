import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import os
import re
import holidays
import networkx as nx
import nltk
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date # Import date for date inputs
import vaderSentiment # Added vaderSentiment import

# Set page configuration
st.set_page_config(page_title="Fraudulent Claim Detection", layout="wide")

# --- Load Model and Preprocessing Components ---

MODEL_PATH = 'best_xgboost_model.pkl'
PREPROCESSOR_PATH = 'fitted_preprocessor.pkl'
TFIDF_PATH = 'fitted_tfidf_vectorizer.pkl'
PCA_PATH = 'fitted_pca_transformer.pkl'
FRAUDULENT_CUSTOMERS_PATH = 'fraudulent_customers.pkl'
CUSTOMER_CLAIM_COUNTS_PATH = 'customer_claim_counts.pkl'
CLAIM_COUNTS_2YEARS_PATH = 'claim_counts_2years.pkl' # Corrected filename case
PERCENTILE_90_PATH = 'percentile_90_claim_amount.pkl'
XTRAIN_COLUMNS_PATH = 'xtrain_columns.pkl'
CUSTOMER_CENTRALITY_PATH = 'customer_centrality_df.pkl'
LOCATION_CENTRALITY_PATH = 'location_centrality_df.pkl'


@st.cache_resource
def load_component(filepath):
    if not os.path.exists(filepath):
        st.error(f"Required file not found: {filepath}. Please ensure all saved preprocessing components and the model file are in the same directory as the app.py file.")
        st.stop()
    try:
        with open(filepath, 'rb') as file:
            component = pickle.load(file)
        return component
    except Exception as e:
        st.error(f"Error loading component from {filepath}: {e}")
        st.stop()


# Load all saved components
model = load_component(MODEL_PATH)
preprocessor = load_component(PREPROCESSOR_PATH)
fitted_tfidf_vectorizer = load_component(TFIDF_PATH)
fitted_pca_transformer = load_component(PCA_PATH)
fraudulent_customers = load_component(FRAUDULENT_CUSTOMERS_PATH)
customer_claim_counts_loaded = load_component(CUSTOMER_CLAIM_COUNTS_PATH)
claim_counts_2years_loaded = load_component(CLAIM_COUNTS_2YEARS_PATH)
percentile_90_claim_amount = load_component(PERCENTILE_90_PATH)
X_train_columns = load_component(XTRAIN_COLUMNS_PATH)
customer_centrality_df_loaded = load_component(CUSTOMER_CENTRALITY_PATH)
location_centrality_df_loaded = load_component(LOCATION_CENTRALITY_PATH)


# Load necessary NLTK data (VADER lexicon)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

download_nltk_data()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Load HuggingFace components for embeddings
@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

embedding_tokenizer, embedding_model = load_embedding_model()

# Define the preprocessing function for new data
def preprocess_data(df):
    """
    Applies the same preprocessing and feature engineering steps as used during training
    to a new DataFrame, using loaded fitted components.

    Args:
        df (pd.DataFrame): DataFrame containing new claim data with original column names.

    Returns:
        pd.DataFrame: The preprocessed data, ready for model prediction, or empty DataFrame if failed.
    """
    df_processed = df.copy()

    # Check if essential columns exist in the input DataFrame
    required_cols = ['Claim_ID', 'Policy_Number', 'Customer_Name', 'Customer_Email',
                     'Customer_Phone', 'Location', 'Policy_Type', 'Claim_Type',
                     'Incident_Type', 'Incident_Date', 'Claim_Submission_Date',
                     'Claim_Amount', 'Claim_Status', 'Adjuster_Notes', 'Customer_Gender',
                     'Customer_Age', 'Customer_Occupation', 'Policy_Start_Date',
                     'Policy_End_Date', 'Premium_Amount'] # Exclude Fraud_Flag as it's the target

    if not all(col in df_processed.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_processed.columns]
        st.error(f"Uploaded file is missing required columns: {', '.join(missing)}")
        return pd.DataFrame()


    # 1. Date Conversion
    date_cols = ['Incident_Date', 'Claim_Submission_Date', 'Policy_Start_Date', 'Policy_End_Date']
    for col in date_cols:
        # Handle potential errors in date conversion
        df_processed[col] = pd.to_datetime(df_processed[col], dayfirst=True, errors='coerce')

    # Drop rows with invalid dates after attempted conversion
    initial_rows = len(df_processed)
    df_processed.dropna(subset=date_cols, inplace=True)
    if len(df_processed) < initial_rows:
         st.warning(f"Dropped {initial_rows - len(df_processed)} rows due to invalid date formats.")

    if df_processed.empty:
        st.error("No valid rows remaining after date parsing.")
        return pd.DataFrame()


    # List to dynamically collect engineered feature names
    engineered_feature_names_list = []

    # 2. Feature Engineering

    # Incident on Holiday
    try:
        years_in_data = df_processed['Incident_Date'].dt.year.unique()
        nigerian_holidays = holidays.Nigeria(years=years_in_data)
        df_processed['Incident_on_Holiday'] = df_processed['Incident_Date'].apply(lambda date: date in nigerian_holidays).astype(int)
        engineered_feature_names_list.append('Incident_on_Holiday')
    except Exception as e:
        st.warning(f"Error engineering 'Incident_on_Holiday': {e}")
        df_processed['Incident_on_Holiday'] = 0 # Ensure column exists
        engineered_feature_names_list.append('Incident_on_Holiday')


    # Claims within Weekends
    try:
        df_processed['Incident_on_Weekend'] = df_processed['Incident_Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        df_processed['Claim_Submission_on_Weekend'] = df_processed['Claim_Submission_Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        engineered_feature_names_list.extend(['Incident_on_Weekend', 'Claim_Submission_on_Weekend'])
    except Exception as e:
        st.warning(f"Error engineering weekend features: {e}")
        df_processed['Incident_on_Weekend'] = 0
        df_processed['Claim_Submission_on_Weekend'] = 0
        engineered_feature_names_list.extend(['Incident_on_Weekend', 'Claim_Submission_on_Weekend'])


    # Delayed Claim submission above 90 days
    try:
        df_processed['Days_to_Claim_Submission'] = (df_processed['Claim_Submission_Date'] - df_processed['Incident_Date']).dt.days
        df_processed['Late_Claim_Submission'] = (df_processed['Days_to_Claim_Submission'] >= 90).astype(int)
        engineered_feature_names_list.extend(['Days_to_Claim_Submission', 'Late_Claim_Submission'])
    except Exception as e:
        st.warning(f"Error engineering claim submission timing features: {e}")
        df_processed['Days_to_Claim_Submission'] = -1 # Default to -1 or a placeholder indicating calculation failed
        df_processed['Late_Claim_Submission'] = 0
        engineered_feature_names_list.extend(['Days_to_Claim_Submission', 'Late_Claim_Submission'])


    # Policy duration
    try:
        df_processed['Policy_Duration_Days'] = (df_processed['Policy_End_Date'] - df_processed['Policy_Start_Date']).dt.days
        engineered_feature_names_list.append('Policy_Duration_Days')
    except Exception as e:
        st.warning(f"Error engineering 'Policy_Duration_Days': {e}")
        df_processed['Policy_Duration_Days'] = -1 # Default
        engineered_feature_names_list.append('Policy_Duration_Days')


    # Claim Count within 2 Years and Frequent Claimant
    # Merge with loaded claim counts within 2 years from training data
    try:
        required_cols_claim_counts_2years = ['Policy_Number', 'Claim_Count_2Years']
        if not all(col in claim_counts_2years_loaded.columns for col in required_cols_claim_counts_2years):
            st.warning(f"Skipping merge for claim counts within 2 years: Loaded data is missing one or more of {required_cols_claim_counts_2years}. Initializing with 0.")
            df_processed['Claim_Count_2Years'] = 0
        else:
            # Ensure merge key 'Policy_Number' is string type in both dataframes
            df_processed['Policy_Number'] = df_processed['Policy_Number'].astype(str)
            claim_counts_2years_loaded['Policy_Number'] = claim_counts_2years_loaded['Policy_Number'].astype(str)
            # Ensure the value column is named correctly in the loaded df before merge
            if 'Claim_Count_2Years' not in claim_counts_2years_loaded.columns:
                 st.warning("Loaded claim_counts_2years_loaded does not have 'Claim_Count_2Years'. Skipping merge.")
                 df_processed['Claim_Count_2Years'] = 0
            else:
                df_processed = df_processed.merge(
                    claim_counts_2years_loaded[['Policy_Number', 'Claim_Count_2Years']],
                    on='Policy_Number',
                    how='left'
                )
                df_processed['Claim_Count_2Years'] = df_processed['Claim_Count_2Years'].fillna(0).astype(int) # Fill NaN and ensure integer type

        df_processed['Frequent_Claimant'] = (df_processed['Claim_Count_2Years'] > 3).astype(int)
        engineered_feature_names_list.extend(['Claim_Count_2Years', 'Frequent_Claimant'])
    except Exception as e:
        st.warning(f"Error engineering claim counts within 2 years: {e}")
        df_processed['Claim_Count_2Years'] = 0
        df_processed['Frequent_Claimant'] = 0
        engineered_feature_names_list.extend(['Claim_Count_2Years', 'Frequent_Claimant'])


    # High Claim amount Flag (using the loaded 90th percentile from training)
    try:
        df_processed['High_Claim_Amount_Flag'] = (df_processed['Claim_Amount'] > percentile_90_claim_amount).astype(int)
        engineered_feature_names_list.append('High_Claim_Amount_Flag')
    except Exception as e:
        st.warning(f"Error engineering 'High_Claim_Amount_Flag': {e}")
        df_processed['High_Claim_Amount_Flag'] = 0
        engineered_feature_names_list.append('High_Claim_Amount_Flag')


    # Calculate the ratio of Claim_Amount to Premium_Amount
    try:
        df_processed['Claim_vs_Premium_Ratio'] = df_processed.apply(lambda row: row['Claim_Amount'] / row['Premium_Amount'] if pd.notna(row['Premium_Amount']) and row['Premium_Amount'] != 0 else 0, axis=1)
        engineered_feature_names_list.append('Claim_vs_Premium_Ratio')
    except Exception as e:
        st.warning(f"Error engineering 'Claim_vs_Premium_Ratio': {e}")
        df_processed['Claim_vs_Premium_Ratio'] = 0.0
        engineered_feature_names_list.append('Claim_vs_Premium_Ratio')


    # Claim Frequency (Claims greater than two times from a particular customer)
    # Merge with loaded customer claim counts from training data
    try:
        required_cols_customer_counts = ['Customer_Name', 'Customer_Claim_Count']
        if not all(col in customer_claim_counts_loaded.columns for col in required_cols_customer_counts):
             st.warning(f"Skipping merge for customer claim counts: Loaded data is missing one or more of {required_cols_customer_counts}. Initializing with 0.")
             df_processed['Customer_Claim_Count'] = 0
        else:
            # Ensure merge key 'Customer_Name' is string type in both dataframes
            df_processed['Customer_Name'] = df_processed['Customer_Name'].astype(str)
            customer_claim_counts_loaded['Customer_Name'] = customer_claim_counts_loaded['Customer_Name'].astype(str)
            # Ensure the value column is named correctly in the loaded df before merge
            if 'Customer_Claim_Count' not in customer_claim_counts_loaded.columns:
                 st.warning("Loaded customer_claim_counts_loaded does not have 'Customer_Claim_Count'. Skipping merge.")
                 df_processed['Customer_Claim_Count'] = 0
            else:
                df_processed = df_processed.merge(
                    customer_claim_counts_loaded[['Customer_Name', 'Customer_Claim_Count']],
                    on='Customer_Name',
                    how='left'
                )
                df_processed['Customer_Claim_Count'] = df_processed['Customer_Claim_Count'].fillna(0).astype(int) # Fill NaN and ensure integer type

        df_processed['Frequent_Customer_Claimant'] = (df_processed['Customer_Claim_Count'] > 2).astype(int)
        engineered_feature_names_list.extend(['Customer_Claim_Count', 'Frequent_Customer_Claimant'])
    except Exception as e:
        st.warning(f"Error engineering customer claim counts: {e}")
        df_processed['Customer_Claim_Count'] = 0
        df_processed['Frequent_Customer_Claimant'] = 0
        engineered_feature_names_list.extend(['Customer_Claim_Count', 'Frequent_Customer_Claimant'])


    # Prior Fraudulent Claim (using the loaded set of fraudulent customers)
    try:
        # Ensure Customer_Name is string type before checking against the set
        df_processed['Customer_Name'] = df_processed['Customer_Name'].astype(str)
        df_processed['Prior_Fraudulent_Claim'] = df_processed['Customer_Name'].apply(lambda x: 1 if x in fraudulent_customers else 0)
        engineered_feature_names_list.append('Prior_Fraudulent_Claim')
    except Exception as e:
        st.warning(f"Error engineering 'Prior_Fraudulent_Claim': {e}")
        df_processed['Prior_Fraudulent_Claim'] = 0
        engineered_feature_names_list.append('Prior_Fraudulent_Claim')


    # Add features for claims within 2 months of policy start/end dates
    try:
        df_processed['Claim_Within_2Months_of_Start'] = ((df_processed['Claim_Submission_Date'] - df_processed['Policy_Start_Date']).dt.days <= 60).astype(int)
        df_processed['Claim_Within_2Months_of_End'] = (((df_processed['Policy_End_Date'] - df_processed['Claim_Submission_Date']).dt.days <= 60) & ((df_processed['Policy_End_Date'] - df_processed['Claim_Submission_Date']).dt.days >= 0)).astype(int)
        engineered_feature_names_list.extend(['Claim_Within_2Months_of_Start', 'Claim_Within_2Months_of_End'])
    except Exception as e:
        st.warning(f"Error engineering policy proximity features: {e}")
        df_processed['Claim_Within_2Months_of_Start'] = 0
        df_processed['Claim_Within_2Months_of_End'] = 0
        engineered_feature_names_list.extend(['Claim_Within_2Months_of_Start', 'Claim_Within_2Months_of_End'])


    # TF-IDF Features
    try:
        tfidf_matrix = fitted_tfidf_vectorizer.transform(df_processed['Adjuster_Notes'].fillna(''))
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=fitted_tfidf_vectorizer.get_feature_names_out())
        # Ensure the index is reset before concat
        df_processed = pd.concat([df_processed.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        engineered_feature_names_list.extend(tfidf_df.columns.tolist()) # Add TF-IDF columns
    except Exception as e:
        st.error(f"Error applying TF-IDF vectorizer: {e}")
        # Add dummy columns based on the fitted vectorizer's output names if TF-IDF fails
        if fitted_tfidf_vectorizer:
             dummy_tfidf_cols = fitted_tfidf_vectorizer.get_feature_names_out().tolist()
             for col in dummy_tfidf_cols:
                  if col not in df_processed.columns:
                       df_processed[col] = 0.0
                       engineered_feature_names_list.append(col)
        else:
             # Fallback if fitted_tfidf_vectorizer is also None/missing
             st.warning("fitted_tfidf_vectorizer not loaded. Cannot add placeholder TF-IDF columns based on fitted names.")
             # Optionally add a fixed number of placeholder columns if needed for preprocessor input consistency
             # for i in range(100): # Example: Add 100 dummy TF-IDF columns
             #      col_name = f"tfidf_placeholder_{i}"
             #      if col_name not in df_processed.columns:
             #           df_processed[col_name] = 0.0
             #           engineered_feature_names_list.append(col_name)


    # Graph-based Features (Customer and Location Centrality)
    try:
        required_cols_centrality = ['Name', 'Centrality_Score']
        merge_successful = True

        # Merge customer centrality
        if not all(col in customer_centrality_df_loaded.columns for col in required_cols_centrality):
            st.warning(f"Skipping merge for customer centrality: Loaded data is missing one or more of {required_cols_centrality}. Initializing with 0.")
            df_processed['Customer_Centrality'] = 0.0
            merge_successful = False
        else:
            df_processed['Customer_Name'] = df_processed['Customer_Name'].astype(str)
            customer_centrality_df_loaded['Name'] = customer_centrality_df_loaded['Name'].astype(str)
            customer_centrality_renamed = customer_centrality_df_loaded.rename(columns={'Centrality_Score': 'Customer_Centrality'})
            df_processed = df_processed.merge(
                customer_centrality_renamed[['Name', 'Customer_Centrality']].rename(columns={'Name': 'Customer_Name'}),
                on='Customer_Name',
                how='left'
            )
            df_processed['Customer_Centrality'].fillna(0.0, inplace=True) # Fill NaN for new customers


        # Merge location centrality
        if not all(col in location_centrality_df_loaded.columns for col in required_cols_centrality):
            st.warning(f"Skipping merge for location centrality: Loaded data is missing one or more of {required_cols_centrality}. Initializing with 0.")
            df_processed['Location_Centrality'] = 0.0
            merge_successful = False
        else:
            df_processed['Location'] = df_processed['Location'].astype(str)
            location_centrality_df_loaded['Name'] = location_centrality_df_loaded['Name'].astype(str)
            location_centrality_renamed = location_centrality_df_loaded.rename(columns={'Centrality_Score': 'Location_Centrality'})
            df_processed = df_processed.merge(
                location_centrality_renamed[['Name', 'Location_Centrality']].rename(columns={'Name': 'Location'}),
                on='Location',
                how='left'
            )
            df_processed['Location_Centrality'].fillna(0.0, inplace=True) # Fill NaN for new locations

        if merge_successful:
             engineered_feature_names_list.extend(['Customer_Centrality', 'Location_Centrality'])
        else:
             # Ensure columns exist even if merge failed
             if 'Customer_Centrality' not in df_processed.columns:
                  df_processed['Customer_Centrality'] = 0.0
                  engineered_feature_names_list.append('Customer_Centrality')
             if 'Location_Centrality' not in df_processed.columns:
                  df_processed['Location_Centrality'] = 0.0
                  engineered_feature_names_list.append('Location_Centrality')


    except Exception as e:
         st.warning(f"Could not apply graph-based features. Ensure centrality data is available and correctly loaded. Error: {e}")
         # Ensure columns exist even on error
         if 'Customer_Centrality' not in df_processed.columns:
            df_processed['Customer_Centrality'] = 0.0
            engineered_feature_names_list.append('Customer_Centrality')
         if 'Location_Centrality' not in df_processed.columns:
            df_processed['Location_Centrality'] = 0.0
            engineered_feature_names_list.append('Location_Centrality')


    # Sentiment Analysis
    try:
        df_processed['Sentiment_Score'] = df_processed['Adjuster_Notes'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
        df_processed['Negative_Tone_Flag'] = (df_processed['Sentiment_Score'] < -0.5).astype(int)
        engineered_feature_names_list.extend(['Sentiment_Score', 'Negative_Tone_Flag'])
    except Exception as e:
        st.warning(f"Error engineering sentiment features: {e}")
        df_processed['Sentiment_Score'] = 0.0
        df_processed['Negative_Tone_Flag'] = 0
        engineered_feature_names_list.extend(['Sentiment_Score', 'Negative_Tone_Flag'])

    # Embedding and PCA Features
    try:
        def get_embedding(text):
            text = str(text)
            # Added return_overflowing_tokens=False to avoid issues
            inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512, return_overflowing_tokens=False)
            with torch.no_grad():
                outputs = embedding_model(**inputs)
            # Handle case where outputs.last_hidden_state is empty or has unexpected shape
            if outputs.last_hidden_state.ndim < 2 or outputs.last_hidden_state.shape[1] == 0:
                 return np.zeros(embedding_model.config.hidden_size) # Return zero vector if embedding fails or is empty
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # Apply embedding function row-wise, handling potential errors
        embedding_results = []
        # Ensure Adjuster_Notes is a string and handle NaNs before applying get_embedding
        for text in df_processed['Adjuster_Notes'].fillna('').astype(str):
             try:
                  embedding_results.append(get_embedding(text))
             except Exception as e:
                  st.warning(f"Error getting embedding for a row: {e}")
                  embedding_results.append(np.zeros(embedding_model.config.hidden_size)) # Append zero vector on error

        embedding_matrix = np.vstack(embedding_results)


        # Apply the fitted PCA transformer
        try:
             reduced_embeddings = fitted_pca_transformer.transform(embedding_matrix)
             embed_pca_feature_names_out = [f'embed_pca_{i+1}' for i in range(fitted_pca_transformer.n_components_)]
             embed_pca_df = pd.DataFrame(reduced_embeddings, columns=embed_pca_feature_names_out)
             # Ensure the index is reset before concat
             df_processed = pd.concat([df_processed.reset_index(drop=True), embed_pca_df.reset_index(drop=True)], axis=1)
             engineered_feature_names_list.extend(embed_pca_df.columns.tolist()) # Add PCA columns
        except Exception as e:
             st.error(f"Error applying PCA transformation: {e}")
             # Add dummy columns based on the fitted transformer's output names if PCA fails
             if fitted_pca_transformer:
                 dummy_pca_cols = [f'embed_pca_{i+1}' for i in range(fitted_pca_transformer.n_components_)]
                 for col in dummy_pca_cols:
                      if col not in df_processed.columns:
                           df_processed[col] = 0.0
                           engineered_feature_names_list.append(col)
             else:
                  # Fallback if fitted_pca_transformer is also None/missing
                  st.warning("fitted_pca_transformer not loaded. Cannot add placeholder PCA columns based on fitted names.")
                  # Optionally add a fixed number of placeholder columns if needed
                  # for i in range(20): # Example: Add 20 dummy PCA columns
                  #      col_name = f"embed_pca_placeholder_{i}"
                  #      if col_name not in df_processed.columns:
                  #           df_processed[col_name] = 0.0
                  #           engineered_feature_names_list.append(col_name)


    except Exception as e:
        st.error(f"Could not engineer Embedding/PCA features. Error: {e}")
        # Adding dummy columns is handled by the except block within the PCA try block
        pass


    # Sentiment Score Scaling - Ensure this column exists and is populated before preprocessor
    # This column was part of the numerical features fed into the preprocessor.
    # We need to ensure it exists before checking for missing preprocessor input columns.
    if 'Sentiment_Score' in df_processed.columns:
        # For now, fill with the raw Sentiment_Score. The preprocessor's scaler should overwrite this.
        # This is a workaround assuming the preprocessor is configured to scale this column.
        df_processed['Sentiment_Score_Scaled'] = df_processed['Sentiment_Score']
        engineered_feature_names_list.append('Sentiment_Score_Scaled')
    else:
        # If Sentiment_Score is also missing, fill with a default numerical value
        df_processed['Sentiment_Score_Scaled'] = 0.0
        engineered_feature_names_list.append('Sentiment_Score_Scaled')
        st.warning("'Sentiment_Score' not found. Filling 'Sentiment_Score_Scaled' with 0.0.")


    # Remove potential duplicates from engineered_feature_names_list
    # This is important if some features were added as fallbacks in except blocks
    engineered_feature_names_list = list(set(engineered_feature_names_list))

    # --- Prepare DataFrame for Preprocessor ---
    # Define the list of columns that should be passed to the preprocessor.
    # This includes base numerical, dynamically collected engineered numerical, and original categorical features.

    # Base numerical features input to the numerical transformer (excluding those already engineered)
    base_numerical_features_input = ['Claim_Amount', 'Customer_Age', 'Premium_Amount']

    # Original categorical features input to the categorical transformer
    original_categorical_features_input = ['Location', 'Policy_Type', 'Claim_Type', 'Incident_Type', 'Customer_Gender', 'Customer_Occupation']

    # Combine base numerical, engineered, and original categorical features for the preprocessor input
    # Ensure order matches the training preprocessor's expected input order.
    # Based on the ColumnTransformer definition, the order is numerical features then categorical features.
    # The numerical features list includes base numerical + engineered numerical + text features + sentiment scaled.

    # Let's assume the order of numerical features input to the num transformer is:
    # ['Claim_Amount', 'Customer_Age', 'Premium_Amount'] + (engineered numerical/binary features) + (TF-IDF features) + (PCA features) + ['Sentiment_Score', 'Negative_Tone_Flag', 'Sentiment_Score_Scaled']
    # This is still an assumption about the exact order used during training.

    # A more robust way is to save the *exact* list of column names that were passed to preprocessor.transform() during training.
    # Since we don't have that, let's try to reconstruct it based on X_train_columns and the preprocessor's output structure.
    # X_train_columns is the output of the preprocessor. The preprocessor's input columns are different.

    # Let's rely on the dynamically collected `engineered_feature_names_list` and the known base/categorical features.
    # The expected input order for the ColumnTransformer is base numerical + engineered features + original categorical.

    # Construct the expected input order list
    expected_preprocessor_input_order = []
    # Add base numerical first
    expected_preprocessor_input_order.extend(base_numerical_features_input)
    # Add engineered features that are not base numerical or original categorical
    expected_preprocessor_input_order.extend([col for col in engineered_feature_names_list if col not in base_numerical_features_input + original_categorical_features_input])
    # Add original categorical last
    expected_preprocessor_input_order.extend(original_categorical_features_input)

    # Ensure all expected input columns are present in df_processed and in the correct order.
    # Add missing columns with default values if they are somehow not present after feature engineering.
    default_values_input = {col: 0.0 for col in expected_preprocessor_input_order}
    for col in original_categorical_features_input:
         default_values_input[col] = 'missing' # Default for categorical

    for col in expected_preprocessor_input_order:
        if col not in df_processed.columns:
            st.warning(f"Final check adding missing preprocessor input column: {col}")
            df_processed[col] = default_values_input.get(col, 0.0) # Add with default value (0.0 fallback)


    # Select and reorder columns to match the expected input order for the preprocessor
    # This DataFrame is what goes into preprocessor.transform()
    # Ensure only columns in expected_preprocessor_input_order are included, in that exact order.
    # Also drop any columns in df_processed that are *not* in expected_preprocessor_input_order.

    # Filter df_processed to keep only the expected input columns and reorder them
    # Use a list comprehension to ensure the order is correct
    df_for_preprocessing = df_processed[[col for col in expected_preprocessor_input_order if col in df_processed.columns]].copy()


    # --- Apply Preprocessing Pipeline (Encoding and Scaling) ---
    try:
        processed_data_array = preprocessor.transform(df_for_preprocessing)

        # Convert the output array back to a DataFrame, using the loaded X_train_columns for column names.
        # This ensures the final output DataFrame has the same columns and order as the training data.
        processed_df = pd.DataFrame(processed_data_array, columns=X_train_columns)

    except Exception as e:
        st.error(f"Error applying preprocessing pipeline: {e}")
        st.write("Please ensure the uploaded file has the expected columns and formats and that all preprocessing components loaded correctly.")
        st.write("Details:", e) # Provide more error details
        return pd.DataFrame()


    return processed_df

# --- Streamlit UI ---

st.title("Fraudulent Claim Detection App")

st.markdown("""
This application allows you to predict whether a claim is fraudulent.
You can either input details for a single claim or upload a CSV file containing multiple claims.
The app will provide a prediction and, for single claims, an explanation using SHAP.
""")

# Option to input single claim or upload dataset
prediction_mode = st.radio("Choose Prediction Mode:", ("Single Claim Input", "Upload Claims Dataset"))

if prediction_mode == "Single Claim Input":
    st.header("Enter Single Claim Details")

    # Input fields for claim details
    col1, col2, col3 = st.columns(3)

    with col1:
        claim_amount = st.number_input("Claim Amount", min_value=0.0, value=100000.0, step=1000.0)
        policy_type = st.selectbox("Policy Type", ['Family', 'Corporate', 'Individual'])
        incident_type = st.selectbox("Incident Type", ['Fire', 'Death', 'Accident', 'Theft', 'Illness'])
        customer_gender = st.selectbox("Customer Gender", ['Female', 'Male'])
        # Ensure default date is a date object, not datetime
        policy_start_date = st.date_input("Policy Start Date", value=date(2023, 1, 1))


    with col2:
        premium_amount = st.number_input("Premium Amount", min_value=0.0, value=50000.0, step=1000.0)
        claim_type = st.selectbox("Claim Type", ['Health', 'Life', 'Auto', 'Fire', 'Gadget'])
        location = st.selectbox("Location", ['Ibadan', 'Port Harcourt', 'Abuja', 'Kano', 'Lagos'])
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
        # Ensure default date is a date object, not datetime
        policy_end_date = st.date_input("Policy End Date", value=date(2025, 7, 13))


    with col3:
        customer_name = st.text_input("Customer Name", value="John Doe")
        customer_email = st.text_input("Customer Email", value="john.doe@example.com")
        customer_phone = st.text_input("Customer Phone", value="(123)456-7890")
        customer_occupation = st.selectbox("Customer Occupation", ['Artisan', 'Unemployed', 'Student', 'Teacher', 'Engineer', 'Driver', 'Trader'])
        # Ensure default date is a date object, not datetime
        incident_date = st.date_input("Incident Date", value=date(2024, 5, 15))
        claim_submission_date = st.date_input("Claim Submission Date", value=date(2024, 5, 20))

    adjuster_notes = st.text_area("Adjuster Notes", value="Claim details seem straightforward. No red flags observed.")

    if st.button("Predict Single Claim"):
        # Create a DataFrame from user input
        new_claim_data = pd.DataFrame({
            'Claim_ID': ['user_input_1'], # Dummy ID
            'Policy_Number': ['USERPOL001'], # Dummy
            'Customer_Name': [customer_name],
            'Customer_Email': [customer_email],
            'Customer_Phone': [customer_phone],
            'Location': [location],
            'Policy_Type': [policy_type],
            'Claim_Type': [claim_type],
            'Incident_Type': [incident_type],
            'Incident_Date': [incident_date.strftime('%d/%m/%Y')], # Convert date objects to string format expected by pd.to_datetime
            'Claim_Submission_Date': [claim_submission_date.strftime('%d/%m/%Y')],
            'Claim_Amount': [claim_amount],
            'Claim_Status': ['Submitted'], # Assume initial status
            'Adjuster_Notes': [adjuster_notes],
            'Customer_Gender': [customer_gender],
            'Customer_Age': [customer_age],
            'Customer_Occupation': [customer_occupation],
            'Policy_Start_Date': [policy_start_date.strftime('%d/%m/%Y')],
            'Policy_End_Date': [policy_end_date.strftime('%d/%m/%Y')],
            'Premium_Amount': [premium_amount],
            'Fraud_Flag': [0] # Target variable (will be predicted) - not used in prediction
        })

        # Preprocess the new claim data
        with st.spinner('Preprocessing data...'):
             preprocessed_new_claim_df = preprocess_data(new_claim_data.copy()) # Use a copy


        if not preprocessed_new_claim_df.empty:
            # Make prediction
            try:
                prediction = model.predict(preprocessed_new_claim_df)
                prediction_proba = model.predict_proba(preprocessed_new_claim_df)[:, 1]

                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    st.error(f"Prediction: Fraudulent Claim")
                    st.write(f"Probability of Fraud: {prediction_proba[0]:.4f}")
                else:
                    st.success(f"Prediction: Legitimate Claim")
                    st.write(f"Probability of Fraud: {prediction_proba[0]:.4f}")

                # --- SHAP Explanation for Single Claim ---
                st.subheader("Explanation (SHAP Force Plot)")
                st.write("Features pushing the prediction higher (towards fraud) are shown in red, those pushing it lower (towards non-fraud) are in blue.")

                # Calculate SHAP values for the single instance
                explainer = shap.TreeExplainer(model)
                # Ensure the input to explainer.shap_values is a DataFrame with the correct column names
                # Need to ensure preprocessed_new_claim_df has the same columns and order as X_train
                # The preprocess_data function is designed to output this using X_train_columns.
                shap_values = explainer.shap_values(preprocessed_new_claim_df)

                # Display the force plot
                shap.initjs()
                # shap.force_plot returns an HTML object
                html_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], preprocessed_new_claim_df.iloc[0,:], show=False)
                st.components.v1.html(html_plot.html(), width=1000, height=300, scrolling=True)

            except Exception as e:
                st.error(f"Error during prediction or SHAP explanation: {e}")
                st.write("Please check the input values and ensure preprocessing steps are correctly aligned with the trained model.")
                st.write("Details:", e)

        else:
             st.warning("Preprocessing failed. Please check the input data.")


elif prediction_mode == "Upload Claims Dataset":
    st.header("Upload Claims Dataset (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded data preview:")
            st.dataframe(df.head())

            # Check for duplicates
            duplicates_count = df.duplicated().sum()
            if duplicates_count > 0:
                st.warning(f"Warning: {duplicates_count} duplicate rows found in the uploaded dataset. These duplicates will be kept for prediction.")
                # Optionally, add a checkbox to allow user to drop duplicates
                # if st.checkbox("Drop duplicate rows?"):
                #     df = df.drop_duplicates().reset_index(drop=True)
                #     st.write(f"Duplicates dropped. Remaining rows: {len(df)}")


            if st.button("Predict Fraud for Dataset"):
                # Preprocess the uploaded dataset
                with st.spinner('Preprocessing data...'):
                    preprocessed_df = preprocess_data(df.copy()) # Use a copy to avoid modifying original df


                if not preprocessed_df.empty:
                    st.write("Preprocessing complete. Making predictions...")

                    # Make predictions for the dataset
                    try:
                        predictions = model.predict(preprocessed_df)
                        prediction_proba = model.predict_proba(preprocessed_df)[:, 1]

                        # Add predictions and probabilities back to the original DataFrame for display
                        df['Predicted_Fraud_Flag'] = predictions
                        df['Fraud_Probability'] = prediction_proba

                        st.subheader("Prediction Results for Uploaded Dataset")
                        st.write("Predictions have been added to the DataFrame:")
                        st.dataframe(df)

                        # Provide a summary
                        fraud_counts = df['Predicted_Fraud_Flag'].value_counts()
                        total_claims = len(df)
                        num_fraudulent = fraud_counts.get(1, 0)
                        num_legitimate = fraud_counts.get(0, 0)

                        st.subheader("Fraud Prediction Summary")
                        st.write(f"Total Claims Processed: {total_claims}")
                        st.write(f"Predicted Fraudulent Claims: {num_fraudulent}")
                        st.write(f"Predicted Legitimate Claims: {num_legitimate}")

                        # Visualize outcomes with a pie chart
                        if total_claims > 0:
                            labels = ['Legitimate', 'Fraudulent']
                            sizes = [num_legitimate, num_fraudulent]
                            colors = ['green', 'red']
                            explode = (0, 0.1)  # explode the 'Fraudulent' slice

                            fig1, ax1 = plt.subplots()
                            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                                    shadow=True, startangle=90)
                            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                            st.pyplot(fig1)


                        # Note: SHAP explanation for a whole dataset is typically done
                        # via summary plots or specific instance selection, not force plots for all.
                        st.subheader("SHAP Explanation Note")
                        st.write("For dataset uploads, detailed SHAP force plots for individual claims are not displayed automatically due to the number of claims. You can use the single claim input mode to get SHAP explanations for specific instances.")


                    except Exception as e:
                         st.error(f"Error during prediction for the dataset: {e}")
                         st.write("Please ensure the uploaded file has the correct columns and data types.")
                         st.write("Details:", e)

                else:
                     st.warning("Preprocessing failed for the uploaded dataset. Please check the file content.")


        except Exception as e:
            st.error(f"Error reading or processing the uploaded file: {e}")
            st.write("Please ensure the file is a valid CSV and matches the expected format.")
            st.write("Details:", e)
