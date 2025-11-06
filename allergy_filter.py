# allergy_filter.py
import spacy
import numpy as np
import pandas as pd
import re

def _batch_vectors(nlp, texts):
    docs = nlp.pipe(texts, batch_size=64, disable=["parser", "tagger", "ner"])
    return [doc.vector if doc.has_vector and np.any(doc.vector) else None for doc in docs]

def _safe_parse_list(val):
    if isinstance(val, str):
        if val.startswith('c(') and val.endswith(')'):
            return re.findall(r'"(.*?)"', val[2:-1])
        try:
            return ast.literal_eval(val)
        except:
            return []
    return val if isinstance(val, list) else []

class EmbeddingAllergyFilter:
    def __init__(self, model_name="en_core_web_md"):
        print("Loading spaCy model …")
        self.nlp = spacy.load(model_name, disable=["parser", "tagger", "ner"])
        print("Model ready.")

    def precompute_dataset_vectors(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["ingredients"] = dataset["RecipeIngredientParts"].apply(_safe_parse_list)
        all_ingredients = [ing for sublist in dataset["ingredients"] for ing in sublist]
        print(f"Computing vectors for {len(all_ingredients):,} ingredients …")
        vectors = _batch_vectors(self.nlp, all_ingredients)
        idx = 0
        grouped = []
        for ing_list in dataset["ingredients"]:
            n = len(ing_list)
            grouped.append([v for v in vectors[idx:idx + n] if v is not None])
            idx += n
        dataset["ingredient_vectors"] = grouped
        return dataset

    def filter_dataset(self, dataset, allergies, threshold=0.85):
        if not self.nlp or not allergies or "ingredient_vectors" not in dataset.columns:
            return dataset
        allergy_vecs = [v for a in allergies if (v := self._get_vector(a)) is not None]
        if not allergy_vecs:
            return dataset
        drop_idx = []
        for idx, vecs in dataset["ingredient_vectors"].items():
            for avec in allergy_vecs:
                norm_a = np.linalg.norm(avec)
                if norm_a == 0: continue
                for ivec in vecs:
                    norm_i = np.linalg.norm(ivec)
                    if norm_i == 0: continue
                    sim = np.dot(avec, ivec) / (norm_a * norm_i)
                    if sim > threshold:
                        drop_idx.append(idx)
                        break
                if idx in drop_idx: break
        return dataset.drop(index=set(drop_idx)) if drop_idx else dataset

    def _get_vector(self, text):
        doc = self.nlp(str(text).lower())
        return doc.vector if doc.has_vector and np.any(doc.vector) else None