#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit web interface for the Find Salem project.
Provides a minimalist photo upload and prediction interface for cat identification.
"""

import os
import streamlit as st
from PIL import Image
from pathlib import Path
from fastai.vision.all import load_learner

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Find Salem",
        page_icon="🐈‍⬛",
        layout="wide"
    )

    st.sidebar.title("🐈‍⬛ Find Salem")
    st.sidebar.caption("Black Cat Identifier")
    page = st.sidebar.radio("", [
        "🔍 Predict",
        "📚 Train Model",
        "📊 Data Management",
        "ℹ️ Model Info"
    ])

    if page == "🔍 Predict":
        predict_page()
    elif page == "📚 Train Model":
        train_page()
    elif page == "📊 Data Management":
        data_management_page()
    elif page == "ℹ️ Model Info":
        model_info_page()

def predict_page():
    # Model loading
    fastai_model_path = "models/salem_fastai_model.pkl"
    if not Path(fastai_model_path).exists():
        st.error("❌ FastAI model not found at 'models/salem_fastai_model.pkl'")
        return

    try:
        model = load_learner(fastai_model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return

    # --- UI Layout ---
    uploaded_files = st.file_uploader("Upload photos for prediction", type=['png', 'jpg', 'jpeg', 'heic'], accept_multiple_files=True)

    st.subheader("Or Select a Test Photo")
    col1, col2 = st.columns(2)
    salem_test_dir = Path("data/test/salem")
    other_test_dir = Path("data/test/other_cats")
    salem_photos = sorted(list(salem_test_dir.glob("*.jpg")))
    other_photos = sorted(list(other_test_dir.glob("*.jpg")))

    with col1:
        selected_salem_photos = st.multiselect("", salem_photos, format_func=lambda p: p.name, placeholder="Select Salem test photos")
    with col2:
        selected_other_photos = st.multiselect("", other_photos, format_func=lambda p: p.name, placeholder="Select Other Cat test photos")

    # --- Prediction Logic ---
    st.subheader("Prediction Results")

    def display_grid(photos_with_labels):
        num_photos = len(photos_with_labels)
        if num_photos == 0:
            return

        # Define the number of columns for the grid
        num_cols = 4
        
        # Iterate through photos in chunks of size `num_cols` to create rows
        for i in range(0, num_photos, num_cols):
            # Create a new row of columns for each chunk
            cols = st.columns(num_cols)
            
            # Get the chunk of photos for the current row
            row_items = photos_with_labels[i:i + num_cols]
            
            # Iterate through the photos and columns for the current row
            for j, (photo, true_label) in enumerate(row_items):
                with cols[j]:
                    image = Image.open(photo)
                    st.image(image, caption=getattr(photo, 'name', 'uploaded file'), use_container_width=True)
                    pred_class, _, probs = model.predict(image)
                    display_pred = "Other Cat" if pred_class == 'other_cats' else "Salem"

                    if true_label:
                        display_text = f"**{true_label} (predicted: {display_pred}, {max(probs):.1%})**"
                        if display_pred == true_label:
                            st.success(display_text)
                        else:
                            st.error(display_text)
                    else:
                        st.info(f"**{display_pred}** ({max(probs):.1%})")

    if uploaded_files:
        # For uploaded files, create list of (photo, None) tuples since we don't know true labels
        photos_with_labels = [(photo, None) for photo in uploaded_files]
        display_grid(photos_with_labels)
    
    # Combine demo photos from both dropdowns into a single list with their labels
    demo_photos_with_labels = []
    demo_photos_with_labels.extend([(photo, "Salem") for photo in selected_salem_photos])
    demo_photos_with_labels.extend([(photo, "Other Cat") for photo in selected_other_photos])
    
    if demo_photos_with_labels:
        display_grid(demo_photos_with_labels)
    
    if not uploaded_files and not selected_salem_photos and not selected_other_photos:
        st.info("Please upload a photo or select a test image to see a prediction.")

def train_page():
    st.header("📚 Train Model")
    st.info("This section is for retraining the model. The current model has 86.7% accuracy.")
    if st.button("Retrain FastAI Model"):
        with st.spinner("Training in progress..."):
            # This would call the training script, e.g., via subprocess
            st.success("Model training script would run here.")

def data_management_page():
    st.header("📊 Data Management")
    st.info("Manage the training and test datasets here.")

def model_info_page():
    st.header("ℹ️ Model Information")
    st.info("Details about the current model and dataset.")
    st.metric("Model Accuracy", "86.7%")


if __name__ == "__main__":
    main()
