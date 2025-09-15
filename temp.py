import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import ast
import re

# --- Allergy Filtering Helper ---

def _filter_dataset(dataset, allergies: list):
    """
    Filters the dataset to exclude recipes containing specified allergens.
    This is now the first step in the recommendation pipeline.
    """
    if not allergies:
        return dataset

    # Create a case-insensitive regex pattern from the allergies list
    # e.g., ['shrimp', 'peanuts'] -> 'shrimp|peanuts'
    allergy_pattern = '|'.join([re.escape(allergen.strip().lower()) for allergen in allergies])

    # Ensure the ingredient parts column is a string before searching
    # We search the raw string representation of the list for efficiency
    mask = dataset['RecipeIngredientParts'].str.lower().str.contains(allergy_pattern, na=False)
    
    # Return a new dataframe containing only the recipes that DO NOT match the allergy pattern
    filtered_df = dataset[~mask]
    
    return filtered_df


# --- HYBRID RECOMMENDATION ENGINE ---

def content_based_recommend(dataset, nutrition_input, params):
    """Generates recipe recommendations based on nutritional similarity."""
    if dataset.empty:
        return pd.DataFrame()

    nutrition_df = dataset.iloc[:, 7:16]
    scaler = StandardScaler()
    scaled_nutrition_df = scaler.fit_transform(nutrition_df)

    nbrs = NearestNeighbors(n_neighbors=params['n_neighbors'], algorithm='brute', metric='cosine').fit(scaled_nutrition_df)
    
    scaled_input = scaler.transform([nutrition_input])
    distances, indices = nbrs.kneighbors(scaled_input)

    # Get the recommendations and their distances
    recommendations_df = dataset.iloc[indices[0]].copy()
    recommendations_df['distance'] = distances[0]
    
    return recommendations_df

def rerank_by_popularity(recommendations_df, popularity_weight=0.3):
    """Re-ranks a dataframe by a weighted popularity score."""
    if recommendations_df.empty:
        return recommendations_df

    # Invert distance to create a similarity score (closer distance = higher similarity)
    recommendations_df['content_score'] = 1 / (recommendations_df['distance'] + 1e-6)
    
    # Normalize scores
    max_content_score = recommendations_df['content_score'].max()
    max_popularity = recommendations_df['popularity'].max()

    if max_content_score > 0:
        recommendations_df['content_score'] /= max_content_score
    if max_popularity > 0:
        recommendations_df['popularity_score'] = recommendations_df['popularity'] / max_popularity
    else:
        recommendations_df['popularity_score'] = 0

    # Calculate hybrid score
    content_weight = 1 - popularity_weight
    recommendations_df['hybrid_score'] = (content_weight * recommendations_df['content_score']) + \
                                       (popularity_weight * recommendations_df['popularity_score'])

    return recommendations_df.sort_values(by='hybrid_score', ascending=False)

def hybrid_recommend(dataset, nutrition_input, params, allergies: list = None):
    """
    Orchestrates the hybrid recommendation process, now with allergy filtering.
    """
    # STAGE 1A: Filter out recipes based on user allergies
    filtered_dataset = _filter_dataset(dataset, allergies)

    if filtered_dataset.empty:
        return pd.DataFrame() # Return empty if no recipes are left after filtering

    # Fetch more neighbors than needed to give the re-ranking model more options
    content_params = params.copy()
    content_params['n_neighbors'] = min(50, len(filtered_dataset)) # Ensure we don't request more neighbors than exist
    
    # 1. Get initial candidates based on content from the filtered dataset
    recommendation_df = content_based_recommend(filtered_dataset, nutrition_input, content_params)

    # 2. Re-rank the candidates using popularity
    reranked_df = rerank_by_popularity(recommendation_df, popularity_weight=0.3)

    # Return the top N final recommendations
    return reranked_df.head(params['n_neighbors'])

# --- Helper function for formatting output ---
def output_recommended_recipes(dataframe):
    """Formats the recommended recipes dataframe into a list of dictionaries."""
    if dataframe is None or dataframe.empty:
        return None
    
    # Safely evaluate string representations of lists
    for col in ['RecipeIngredientParts', 'RecipeInstructions']:
        dataframe[col] = dataframe[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    return dataframe.to_dict(orient='records')

