import streamlit as st

st.title("🎯 My First Auto-Deployed App--Test")
st.write("From VSCode to Cloud in 3 Steps!")

name = st.text_input("Enter your name:")
if name:
  st.success(f"Hello {name}! App deployed successfully!")