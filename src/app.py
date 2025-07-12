import streamlit as st

input_dataFrame = st.Page("page1.py", title="imput dataFrame", icon="📊")
compare_methods = st.Page("page2.py", title="compare methods", icon="📈")
pg = st.navigation([input_dataFrame, compare_methods])
st.set_page_config(page_title="Imputation of missing data app", page_icon="💧")
pg.run()
