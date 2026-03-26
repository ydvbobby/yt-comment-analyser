import requests
import streamlit as st


url = "http://127.0.0.1:8000/predict"

user_input = st.text_area("Enter your comment for analysis:")
data = {
    "text": [str(user_input)]
}
if st.button("Analyze Comment"):
    try:
        response = requests.post(url=url, json=data)
        if response.status_code == 200:
            result = response.json()['predictions'][0]
            if result == 1:
                st.success("The comment is positive.")
            elif result == 0:
                st.warning("The comment is Non-Toxic.")
            else: 
                st.error("This is toxic comment.")
    except:
        st.write('Server Busy')




