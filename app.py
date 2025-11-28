#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import tensorflow as tf
import pickle
import re

# --- Load model & tokenizer ---
model = tf.keras.models.load_model("rnn_lstm_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 256

# --- Text preprocessing ---
def normalize(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection with RNN-LSTM")
st.write("Paste any news article or headline to check if it’s **Fake** or **Real**.")

user_input = st.text_area("Enter news text here:")

if st.button("Check"):
    if user_input.strip() != "":
        cleaned_text = normalize(user_input)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, padding='post', maxlen=maxlen)
        prediction = model.predict(padded)[0][0]

        if prediction >= 0.5:
            st.success("✅ This looks like **Real News**.")
        else:
            st.error("🚨 This looks like **Fake News**.")
    else:
        st.warning("⚠️ Please enter some text first.")


# In[ ]:




