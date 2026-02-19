import streamlit as st

pg = st.navigation({
    "HW Manager": [
        st.Page("HW/HW1.py", title="HW 1"),
        st.Page("HW/HW2.py", title="HW 2"),
        st.Page("HW/HW3.py", title="HW 3"),
        st.Page("HW/HW4.py", title="HW 4"),
        st.Page("HW/HW5.py", title="HW 5", default=True),
        st.Page("HW/HW6.py", title="HW 6"),
        st.Page("HW/HW7.py", title="HW 7"),
        ]
    }
)    
pg.run()