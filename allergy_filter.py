import spacy
import numpy as np
import ast
import pandas as pd

# --- NEW: A Robust Function to Safely Parse String Lists ---
def _safe_literal_eval(val):
    """
    Safely evaluates a string that should be a Python list.
    If the string is malformed, empty, or not a string, it returns an empty list.
    """
    if isinstance(val, str):
        try:
            # Attempt to parse the string into a Python object (a list)
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            # If parsing fails, it's a malformed string. Return an empty list.
            return []
    elif isinstance(val, list):
        # If it's already a list, just return it
        return val
    else:
        # For any other types (like NaN, None), return an empty list
        return []

class EmbeddingAllergyFilter:
    """
    Uses word embeddings to intelligently filter recipes based on
    the semantic similarity of ingredients to known allergens.
    """
    def __init__(self, model_name="en_core_web_md"):
        print("Initializing Allergy Filter: Loading NLP model...")
        try:
            self.nlp = spacy.load(model_name)
            print("NLP model loaded successfully.")
        except OSError:
            print(f"--- FATAL ERROR ---")
            print(f"spaCy model '{model_name}' not found.")
            print(f"Please run this command in your terminal to download it:")
            print(f"python -m spacy download {model_name}")
            print(f"-------------------")
            self.nlp = None

    def _get_ingredient_vectors(self, ingredient_list: list) -> list:
        if not self.nlp or not ingredient_list:
            return []
        
        vectors = []
        for ingredient in ingredient_list:
            doc = self.nlp(str(ingredient).lower()) # Ensure ingredient is a string
            if doc.has_vector and doc.vector.any():
                vectors.append(doc.vector)
        return vectors

    def precompute_dataset_vectors(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if not self.nlp:
            print("Cannot precompute vectors: NLP model not loaded.")
            dataset['ingredient_vectors'] = [[] for _ in range(len(dataset))]
            return dataset

        print("Pre-computing ingredient vectors for the entire dataset...")
        
        # --- FIXED: Use the new safe function instead of a lambda ---
        dataset['RecipeIngredientParts_list'] = dataset['RecipeIngredientParts'].apply(_safe_literal_eval)
        
        dataset['ingredient_vectors'] = dataset['RecipeIngredientParts_list'].apply(self._get_ingredient_vectors)
        
        print("Ingredient vector pre-computation complete.")
        return dataset

    def filter_dataset(self, dataset: pd.DataFrame, allergies: list, similarity_threshold=0.85) -> pd.DataFrame:
        if not allergies or not self.nlp:
            return dataset

        print(f"Filtering dataset for allergies: {allergies}")
        allergy_vectors = self._get_ingredient_vectors(allergies)
        
        if not allergy_vectors:
            print("Warning: Could not generate vectors for the provided allergies.")
            return dataset

        indices_to_drop = []
        for index, row in dataset.iterrows():
            is_allergic_recipe = False
            for allergy_vec in allergy_vectors:
                for ingredient_vec in row.get('ingredient_vectors', []):
                    dot_product = np.dot(allergy_vec, ingredient_vec)
                    norm_allergy = np.linalg.norm(allergy_vec)
                    norm_ingredient = np.linalg.norm(ingredient_vec)
                    
                    if norm_allergy > 0 and norm_ingredient > 0:
                        similarity = dot_product / (norm_allergy * norm_ingredient)
                        
                        if similarity > similarity_threshold:
                            indices_to_drop.append(index)
                            is_allergic_recipe = True
                            break
            
                if is_allergic_recipe:
                    break
                
        if indices_to_drop:
            unique_indices = list(set(indices_to_drop))
            print(f"Found {len(unique_indices)} recipes to remove based on allergies.")
            return dataset.drop(index=unique_indices)
        else:
            print("No recipes found containing the specified allergens.")
            return dataset

