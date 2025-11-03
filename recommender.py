import pandas as pd
import numpy as np
import ast

# --- A Robust Function to Safely Parse String Lists ---
def _safe_literal_eval(val):
    """
    Safely evaluates a string that should be a Python list.
    If the string is malformed, empty, NaN, or not a string, it returns an empty list.
    This prevents the application from crashing on messy data.
    """
    if isinstance(val, str):
        try:
            # Attempt to parse the string into a Python object (e.g., a list)
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            # If parsing fails, it's a malformed string.
            return []
    elif isinstance(val, list):
        # If it's already a list, it's safe to return.
        return val
    else:
        # For any other types (like NaN, None), return an empty list as a safe default.
        return []

# --- From-Scratch Machine Learning Components (Vectorized for Speed) ---

def manual_standard_scaler(df, input_vector):
    """
    Manually standardizes a DataFrame and an input vector using efficient, vectorized operations.
    This function replaces scikit-learn's StandardScaler.
    """
    # Calculate mean and standard deviation for each column
    means = df.mean(axis=0)
    stds = df.std(axis=0)
    
    # To prevent division by zero if a column has zero variance
    stds[stds == 0] = 1
    
    # Apply the Z-score formula to the entire DataFrame at once
    scaled_df = (df - means) / stds
    
    # Apply the Z-score formula to the user's input vector
    scaled_vector = (np.array(input_vector) - means.values) / stds.values
    
    return scaled_df, scaled_vector

def manual_nearest_neighbors(scaled_df, scaled_vector):
    """
    Manually finds the single nearest neighbor using vectorized cosine similarity.
    This function replaces scikit-learn's NearestNeighbors.
    """
    # Calculate the dot product of every recipe vector with the user's vector
    dot_product = np.dot(scaled_df.values, scaled_vector)
    
    # Calculate the magnitude (L2 norm) of every recipe vector
    dataset_norm = np.linalg.norm(scaled_df.values, axis=1)
    
    # Calculate the magnitude of the user's vector
    vector_norm = np.linalg.norm(scaled_vector)
    
    # Calculate the cosine similarity for all recipes at once
    # Adding a small epsilon (1e-9) to prevent division by zero
    similarities = dot_product / (dataset_norm * vector_norm + 1e-9)
    
    # Find the index of the recipe with the highest similarity score
    best_match_index = np.argmax(similarities)
    
    return best_match_index

# --- Main Recommendation Function ---

def content_based_recommend(dataset, nutrition_input):
    """
    Generates a single best recipe recommendation by orchestrating the from-scratch components.
    """
    if dataset.empty:
        return pd.DataFrame()

    # Define and select the nutritional columns by name for robustness
    nutrition_columns = [
        'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 
        'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
    ]
    nutrition_df = dataset[nutrition_columns].astype(float)
    
    # 1. Manually scale the data using our custom function
    scaled_nutrition_df, scaled_input_vector = manual_standard_scaler(nutrition_df, nutrition_input)

    # 2. Manually find the single nearest neighbor
    best_index = manual_nearest_neighbors(
        scaled_nutrition_df, 
        scaled_input_vector
    )

    # Return the single best recipe as a DataFrame
    return dataset.iloc[[best_index]]

# --- Helper function for formatting the final API output ---

def output_recommended_recipes(dataframe):
    """
    Formats the recommended recipes dataframe into a list of dictionaries for the API response.
    """
    if dataframe is None or dataframe.empty:
        return None
    
    output_df = dataframe.copy()
    
    # Use the safe parsing function to handle potentially messy data before outputting
    for col in ['RecipeIngredientParts', 'RecipeInstructions']:
        output_df[col] = output_df[col].apply(_safe_literal_eval)
    
    return output_df.to_dict(orient='records')
