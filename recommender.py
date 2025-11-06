# recommender.py
import pandas as pd
import numpy as np
import ast
import re

def _safe_parse_list(val):
    if isinstance(val, str):
        if val.startswith('c(') and val.endswith(')'):
            return re.findall(r'"(.*?)"', val[2:-1])
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(val, list):
        return val
    else:
        return []

def manual_standard_scaler(df, input_vector):
    means = df.mean(axis=0)
    stds = df.std(axis=0)
    stds[stds == 0] = 1
    scaled_df = (df - means) / stds
    scaled_vector = (np.array(input_vector) - means.values) / stds.values
    return scaled_df, scaled_vector

def manual_nearest_neighbors(scaled_df, scaled_vector):
    dot = np.dot(scaled_df.values, scaled_vector)
    norm_df = np.linalg.norm(scaled_df.values, axis=1)
    norm_vec = np.linalg.norm(scaled_vector)
    similarities = dot / (norm_df * norm_vec + 1e-9)
    return np.argmax(similarities)

def content_based_recommend(dataset, nutrition_input):
    if dataset.empty: return pd.DataFrame()
    cols = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
            'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']
    nutrition_df = dataset[cols].astype(float)
    scaled_df, scaled_vec = manual_standard_scaler(nutrition_df, nutrition_input)
    idx = manual_nearest_neighbors(scaled_df, scaled_vec)
    return dataset.iloc[[idx]]

def output_recommended_recipes(df):
    if df is None or df.empty: return None
    df = df.copy()
    for col in ['RecipeIngredientParts', 'RecipeInstructions']:
        df[col] = df[col].apply(_safe_parse_list)
    return df.to_dict('records')