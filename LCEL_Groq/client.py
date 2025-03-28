import requests
import streamlit as st


def get_groq_response(input_text):
  json_body={
    "input": {
      "language": "French",
      "text": input_text
    },
    "config": {},
    "kwargs": {}
  }
  response=requests.post("http://127.0.0.1:8000/chain/invoke",json = json_body)
  print(response.json())
  return response.json()

## Streamlit app
st.title("LLM Application Using LCEL")
input_text=st.text_input("Enter the text you want to convert to french")
print(input_text, type(input_text))

if input_text:
  st.write(get_groq_response(input_text))