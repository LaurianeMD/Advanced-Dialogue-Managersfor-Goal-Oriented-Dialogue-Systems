import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Charger le modèle et le tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fonction pour générer une réponse
def generate_response(user_input):
    # Tokeniser l'entrée utilisateur
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    # Générer la réponse du modèle
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Décoder la réponse
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Interface utilisateur Streamlit
st.title("Chatbot Orienté Objectifs")
st.write("Posez vos questions au chatbot!")

# Entrée utilisateur
user_input = st.text_input("Vous:")

if user_input:
    # Générer la réponse du chatbot
    response = generate_response(user_input)
    st.write(f"Chatbot: {response}")
