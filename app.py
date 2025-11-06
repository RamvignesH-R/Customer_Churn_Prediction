import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import numpy as np

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from profiler import calculateNutritionalTarget
from recommender import _safe_parse_list
from allergy_filter import EmbeddingAllergyFilter

st.set_page_config(page_title="NutriMatch AI", page_icon="apple", layout="wide")


st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0;}
    .card {background: rgba(255,255,255,0.08); backdrop-filter: blur(12px);
           border-radius: 16px; border: 1px solid rgba(255,255,255,0.12);
           padding: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.2); margin-bottom: 1.5rem;}
    .title {font-size: 2.8rem; font-weight: 800; text-align: center;
            background: linear-gradient(90deg, #60a5fa, #34d399);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: .5rem;}
    .subtitle {text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;}
    .metric-box {background: rgba(255,255,255,0.1); border-radius: 12px;
                 padding: .8rem 1.2rem; text-align: center; margin: .4rem;
                 font-weight: 600; border: 1px solid rgba(255,255,255,0.15);}
    .stButton > button {background: linear-gradient(45deg, #3b82f6, #10b981);
                        color: white; border: none; border-radius: 12px;
                        padding: .7rem 1.5rem; font-weight: 600; width: 100%;}
    .stButton > button:hover {transform: translateY(-2px);
                              box-shadow: 0 6px 20px rgba(59,130,246,0.4);}
    .ingredient {background: rgba(255,255,255,0.08); padding: .4rem .8rem;
                 border-radius: 8px; margin: .3rem 0; font-size: .95rem;}
    .stSelectbox > div > div {background: rgba(255,255,255,0.1); border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing AI engine...")
def load_resources():
    data_path = BASE_DIR / "data.csv"
    if not data_path.exists():
        st.error("`data.csv` not found in project directory.")
        st.stop()

    with st.spinner("Loading 2,000+ recipes & building AI brain..."):
        df = pd.read_csv(data_path)
        filterer = EmbeddingAllergyFilter()
        if not filterer.nlp:
            st.stop()
        df = filterer.precompute_dataset_vectors(df)
    st.success("AI Ready – Instant Recommendations!")
    return df, filterer

dataset, allergy_filter = load_resources()

common_allergies = [
    'Milk', 'Eggs', 'Peanuts', 'Tree Nuts', 'Soy', 'Wheat',
    'Fish', 'Shellfish', 'Sesame', 'Mustard', 'Sulfites', 'Other'
]


with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/artificial-intelligence.png", width=80)
    st.markdown("## Your Profile")

    age = st.slider("Age", 1, 100, 30)
    gender = st.radio("Gender", ["male", "female"])
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, 0.5)
    height = st.number_input("Height (cm)", 100.0, 250.0, 170.0, 0.5)
    activity = st.selectbox("Activity Level", ["sedentary", "light", "moderate", "active"])
    goal = st.selectbox("Goal", ["lose", "maintain", "gain"])
    sugar = st.selectbox("Blood Sugar", ["normal", "high"])

    selected_allergies = st.multiselect("Allergies", common_allergies, default=[])

    other_allergies = ""
    if 'Other' in selected_allergies:
        other_allergies = st.text_input("Other Allergies (comma-separated)", placeholder="e.g., gluten, latex")

    allergies = [a.lower() for a in selected_allergies if a != 'Other']
    if other_allergies:
        allergies.extend([a.strip().lower() for a in other_allergies.split(",") if a.strip()])

    if st.button("Find Safe Recipes", type="primary"):
        st.session_state.profile = {
            "age": age, "gender": gender, "weight": weight, "height": height,
            "activity": activity, "goal": goal, "sugar": sugar, "allergies": allergies
        }

        if "safe_df" in st.session_state:
            del st.session_state.safe_df
        if "target" in st.session_state:
            del st.session_state.target

if "profile" in st.session_state:
    p = st.session_state.profile


    if "target" not in st.session_state:
        st.session_state.target = calculateNutritionalTarget(
            p["weight"], p["height"], p["age"], p["gender"],
            p["activity"], p["goal"], p["sugar"]
        )
    target = st.session_state.target
    labels = ["Calories", "Fat", "Sat Fat", "Chol", "Sodium", "Carbs", "Fiber", "Sugar", "Protein"]


    if "safe_df" not in st.session_state:
        with st.spinner("Filtering safe recipes..."):
            filtered = allergy_filter.filter_dataset(dataset.copy(), p["allergies"])
            st.session_state.safe_df = filtered if filtered is not None else pd.DataFrame()
    safe_df = st.session_state.safe_df


    if not isinstance(safe_df, pd.DataFrame):
        safe_df = pd.DataFrame()
        st.session_state.safe_df = safe_df

    st.markdown(f"### **{len(safe_df):,} safe recipes found**")

    if safe_df.empty:
        st.warning("No safe recipes found. Try removing some allergies.")
        st.stop()


    cols = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
            'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']
    nutrition_df = safe_df[cols].astype(float)
    means = nutrition_df.mean(axis=0)
    stds = nutrition_df.std(axis=0)
    stds[stds == 0] = 1
    scaled_df = (nutrition_df - means) / stds
    scaled_vec = (np.array(target) - means.values) / stds.values
    dot = np.dot(scaled_df.values, scaled_vec)
    norm_df = np.linalg.norm(scaled_df.values, axis=1)
    norm_vec = np.linalg.norm(scaled_vec)
    similarities = dot / (norm_df * norm_vec + 1e-9)

    top_indices = np.argsort(similarities)[::-1][:50]
    top_df = safe_df.iloc[top_indices].copy()
    top_df['Similarity'] = similarities[top_indices]
    recipe_options = top_df['Name'].tolist()


    selected_name = st.selectbox("Choose a Recipe", recipe_options, index=0)

    if selected_name:
        row = top_df[top_df['Name'] == selected_name].iloc[0]
        result = {
            'Name': row['Name'],
            'TotalTime': row['TotalTime'],
            'Calories': row['Calories'],
            'FatContent': row['FatContent'],
            'SaturatedFatContent': row['SaturatedFatContent'],
            'CholesterolContent': row['CholesterolContent']/1000,
            'SodiumContent': row['SodiumContent']/1000,
            'CarbohydrateContent': row['CarbohydrateContent'],
            'FiberContent': row['FiberContent'],
            'SugarContent': row['SugarContent'],
            'ProteinContent': row['ProteinContent'],
            'RecipeIngredientParts': _safe_parse_list(row.get('RecipeIngredientParts', '')),
            'RecipeInstructions': _safe_parse_list(row.get('RecipeInstructions', ''))
        }
        

        col1, col2 = st.columns([1, 1.3])

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Your Daily Target")
            for k, v in zip(labels, target):
                unit = "kcal" if k == "Calories" else "mg" if k in ["Chol", "Sodium"] else "g"
                st.markdown(f"<div class='metric-box'>{k}: <strong>{v:,.0f}</strong> {unit}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            fig_target = go.Figure(go.Pie(
                labels=labels[1:], values=target[1:], hole=0.5,
                textinfo='label+percent', marker=dict(colors=px.colors.sequential.Teal),
                hoverinfo='label+value'
            ))
            fig_target.update_layout(title="Your Daily Target", title_x=0.5, showlegend=False, height=380)

            rec_vals = [(result[k]/1000) if (k=='SodiumContent' or k=='CholesterolContent') else result[k] for k in [
                'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent',
                'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
            ]]
            fig_recipe = go.Figure(go.Pie(
                labels=labels[1:], values=rec_vals, hole=0.5,
                textinfo='label+percent', marker=dict(colors=px.colors.sequential.Emrld),
                hoverinfo='label+value'
            ))
            fig_recipe.update_layout(title=result['Name'], title_x=0.5, showlegend=False, height=380)

            st.plotly_chart(fig_target, use_container_width=True)
            st.plotly_chart(fig_recipe, use_container_width=True)


        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {result['Name']}")
        st.markdown(f"**Total Time:** {result['TotalTime']} min")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-box'>Protein<br><strong>{result['ProteinContent']:.0f}g</strong></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-box'>Carbs<br><strong>{result['CarbohydrateContent']:.0f}g</strong></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-box'>Fat<br><strong>{result['FatContent']:.0f}g</strong></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-box'>Calories<br><strong>{result['Calories']:.0f}</strong></div>", unsafe_allow_html=True)

        col_ing, col_inst = st.columns([1, 1])
        with col_ing:
            st.markdown("**Ingredients**")
            ingredients = result['RecipeIngredientParts']
            for ing in ingredients[:10]:
                st.markdown(f"<div class='ingredient'>• {ing}</div>", unsafe_allow_html=True)
            if len(ingredients) > 10:
                with st.expander("Show all ingredients"):
                    for ing in ingredients[10:]:
                        st.markdown(f"• {ing}")

        with col_inst:
            st.markdown("**Instructions**")
            instructions = result['RecipeInstructions']
            for i, step in enumerate(instructions[:5], 1):
                st.markdown(f"{i}. {step}")
            if len(instructions) > 5:
                with st.expander("View full recipe"):
                    for i, step in enumerate(instructions, 1):
                        st.markdown(f"{i}. {step}")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<h1 class='title'>NutriMatch AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Your Personal AI Nutritionist – Safe, Smart, Delicious</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; margin-top:2rem;'>
        <p style='font-size:1.1rem; color:#94a3b8;'>
            Enter your profile on the left to get a <strong>personalized, allergy-safe recipe</strong> in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)