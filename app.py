import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load laptop dataset
final_df = pd.read_csv("laptops.csv")

# Load model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="AI Laptop Finder", page_icon="💻")

@st.cache_resource
def create_index():
    descriptions = final_df['Model'].astype(str).tolist()  # Changed from 'description'
    embeddings = model.encode(descriptions, show_progress_bar=True)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = create_index()

# Semantic search
def semantic_search(query, top_k=10):
    query_embedding = model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    results = final_df.iloc[indices[0]]
    return results

# Streamlit UI
st.title("💻 AI-Powered Laptop Search")
st.markdown("Ask for laptops by budget, processor, RAM, or use case (e.g., gaming, office, editing).")

query = st.text_input("What are you looking for?", placeholder="e.g., i7 laptop for video editing under 60000")

if query:
    with st.spinner("Searching..."):
        results = semantic_search(query)

        # 💰 Price Filter
        min_price = int(final_df["Price"].min())  # Changed from 'price'
        max_price = int(final_df["Price"].max())
        selected_min, selected_max = st.slider("💰 Price Range", min_price, max_price, (min_price, max_price), step=1000)

        # ⭐ Rating Filter
        if "Rating" in final_df.columns:  # Changed from 'rating'
            min_rating = st.selectbox("⭐ Minimum Rating", [0, 1, 2, 3, 4, 5], index=3)
            results = results[
                (results["Price"] >= selected_min) &
                (results["Price"] <= selected_max) &
                (results["Rating"] >= min_rating)
            ]
        else:
            results = results[
                (results["Price"] >= selected_min) &
                (results["Price"] <= selected_max)
            ]

        # Display filtered results
        if not results.empty:
            st.success("Here are laptops that match your needs:")
            st.write(results[['brand', 'Model', 'Price', 'processor_brand', 'ram_memory']])
            # st.write(results[['brand', 'Model', 'Price', 'processor_brand', 'ram_memory']])  # Adjusted columns
        else:
            st.warning("No laptops found matching your criteria.")

# Chatbot Section
st.markdown("---")
st.title("🤖 Chat with the Laptop Assistant")
user_input = st.text_input("You:", key="chat_input")

# # Groq API Key
GROQ_API_KEY = st.secrets.get("Enter Your API KEY") 

if user_input:
    with st.spinner("Thinking..."):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that recommends laptops based on user needs."},
                    {"role": "user", "content": user_input}
                ]
            )
            reply = response.choices[0].message.content
            st.markdown("**Assistant:** " + reply)
        except Exception as e:
            st.error(f"Error: {e}")

