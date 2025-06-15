import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from neo4j import GraphDatabase

# ------------------- Setup -------------------

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    train_df = pd.read_csv("small_train_data.csv")
    test_df = pd.read_csv("small_test_data.csv")

    train_df.dropna(subset=["uniq_id", "product_name", "main_category", "price", "average_review_rating"], inplace=True)
    test_df.dropna(subset=["uniq_id", "product_name", "main_category", "price", "average_review_rating"], inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    return train_df, test_df

@st.cache_resource
def get_driver():
    return GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "66778899"))

# ------------------- Neo4j Query -------------------

def get_ontology_based_ids(product_id, top_k=10):
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Product {id: $product_id})
            OPTIONAL MATCH (p)-[:SAME_CATEGORY]-(cat:Product)
            WITH p, collect(DISTINCT cat) AS cats
            OPTIONAL MATCH (p)-[:SAME_SUB_CATEGORY]-(sub:Product)
            WITH p, cats, collect(DISTINCT sub) AS subs
            OPTIONAL MATCH (p)-[:SAME_MANUFACTURER]-(manu:Product)
            WITH p, cats, subs, collect(DISTINCT manu) AS manus
            OPTIONAL MATCH (p)-[:SIMILAR_PRICE]-(price:Product)
            WITH p, cats, subs, manus, collect(DISTINCT price) AS prices
            OPTIONAL MATCH (p)-[:SIMILAR_RATING]-(rate:Product)
            WITH p, cats, subs, manus, prices, collect(DISTINCT rate) AS rates
            WITH cats + subs + manus + prices + rates AS all_recs, p.id AS targetId
            UNWIND all_recs AS rec
            WITH rec.id AS id, count(*) AS score, targetId
            WHERE id IS NOT NULL AND id <> targetId
            RETURN id ORDER BY score DESC LIMIT $top_k
        """, {"product_id": product_id, "top_k": top_k})
        return [record["id"] for record in result]

# ------------------- Recommendation Logic -------------------

def search_and_recommend(user_input, top_k=5):
    user_emb = model.encode(user_input, convert_to_tensor=True)

    exact_match = train_df[train_df["product_name"].str.strip().str.lower() == user_input.strip().lower()]
    if not exact_match.empty:
        closest = exact_match.iloc[0]
    else:
        train_df["sim"] = train_df["embedding"].apply(lambda x: util.pytorch_cos_sim(user_emb, x).item())
        closest = train_df.sort_values(by="sim", ascending=False).iloc[0]

    st.subheader("🔍 Most similar product found:")
    st.write(f"**Product Name:** {closest['product_name']}")
    st.write(f"**Category:** {closest['main_category']}")
    st.write(f"**Price:** RM{closest['price']}")
    st.write(f"**Rating:** {closest['average_review_rating']}")

    ontology_ids = get_ontology_based_ids(closest["uniq_id"], top_k=50)
    recs_df = train_df[train_df["uniq_id"].isin(ontology_ids)].copy()
    recs_df["sim"] = recs_df["embedding"].apply(lambda x: util.pytorch_cos_sim(user_emb, x).item())

    negatives = train_df[~train_df["uniq_id"].isin(ontology_ids)].sample(n=min(50, len(train_df)), random_state=42)
    negatives["sim"] = negatives["embedding"].apply(lambda x: util.pytorch_cos_sim(user_emb, x).item())

    combined_df = pd.concat([recs_df.assign(true=1), negatives.assign(true=0)])
    combined_df.sort_values(by="sim", ascending=False, inplace=True)

    threshold = combined_df["sim"].mean()
    combined_df["pred"] = combined_df["sim"].apply(lambda x: 1 if x >= threshold else 0)

    y_true = combined_df["true"]
    y_pred = combined_df["pred"]
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    st.subheader("🎯 Top 5 Recommended Products:")
    st.dataframe(recs_df.sort_values(by="sim", ascending=False).head(top_k)[["product_name", "main_category", "price", "average_review_rating"]])

    st.subheader("📊 Evaluation Metrics")
    st.text(f"Precision:  {precision:.4f}")
    st.text(f"Recall:     {recall:.4f}")
    st.text(f"F1-Score:   {f1:.4f}")
    st.text(f"Accuracy:   {accuracy:.4f}")

# ------------------- Streamlit UI -------------------

st.title("🛒 Ontology-Based Recommender System")

# Load once
model = load_model()
train_df, test_df = load_data()
train_df["embedding"] = train_df["product_name"].apply(lambda x: model.encode(str(x), convert_to_tensor=True))
driver = get_driver()

# UI Section
st.markdown("### Choose a product from dropdown or type manually:")

product_list = sorted(train_df["product_name"].unique().tolist())
col1, col2 = st.columns([3, 2])
with col1:
    selected_product = st.selectbox("Select a product:", ["-- None --"] + product_list)

with col2:
    manual_input = st.text_input("Or enter a product name to search:")

# Final input
final_input = manual_input.strip() if manual_input else (selected_product if selected_product != "-- None --" else "")

if final_input:
    search_and_recommend(final_input)
