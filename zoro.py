import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random

with open("zoropersonality.json", "r") as file:
    zoropersonality= json.load(file)  #load personality

tokenizer= AutoTokenizer.from_pretrained("PygmalionAI/metharme-1.3b")
model= AutoModelForCausalLM.from_pretrained("PygmalionAI/metharme-1.3b")  #metharme

scenario = "On a dark alley you are being attacked by thieves. Suddenly, Zoro comes to your rescue. What do you do?"

st.title("Adventures of Zoro")
st.write("Scenario:", scenario)
user_input = st.text_input("Your response to Zoro:")

if user_input:
    chosen_trait =random.choice(zoropersonality["personality_traits"])
    explanation = f"Zoro is {chosen_trait}."
    context =f"{explanation} Scenario: {scenario}\nYou said: {user_input}"
    input_ids= tokenizer.encode(context, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  #creativity
        top_k= 50,  #remove long tail low probability responses for diversity
        repetition_penalty= 1.2,  #penalize new tokens based on whether they appear in the text so far (increase new topic)
        do_sample= True
    )

    response= tokenizer.decode(output[0], skip_special_tokens=True)
    st.write("Zoro's Response:", response)
