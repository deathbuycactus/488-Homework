import streamlit as st

pg = st.navigation({
    "HW Manager": [
        st.Page("HW1.py", title="HW 1"),
        st.Page("HW2.py", title="HW 2", default=True),
        st.Page("HW3.py", title="HW 3"),
        st.Page("HW4.py", title="HW 4"),
        st.Page("HW5.py", title="HW 5"),
        st.Page("HW6.py", title="HW 6"),
        st.Page("HW7.py", title="HW 7"),
        st.Page("HW8.py", title="HW 8"),
        st.Page("HW9.py", title="HW 9"),
        ]
    }
)    
pg.run()