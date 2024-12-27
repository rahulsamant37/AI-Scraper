import streamlit as st
from streamlit_tags import st_tags
import base64


    

def main():
    st.set_page_config(
        page_title="Universal Web Scraper",
        page_icon="🦑",
        layout="wide"
    )
    st.markdown("""
        <style>
            .stApp {
                background-color: #1c1e22;
                color: #ffffff;
            }
            /* Main content area background and text color */
            [data-testid="stHeader"] {
                background-color: #1c1e22;
                color: #ffffff;    
            }
            /* Sidebar background and text color */
            [data-testid="stSidebar"] {
                background-color: #22272e;
                color: #ffffff;
            }
            /* Set all text elements inside the sidebar to white */
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
                color: #ffffff;
            }
            /* Sidebar input fields background and text color */
            [data-testid="stSidebar"] input, [data-testid="stSidebar"] select,
            [data-testid="stSidebar"] textarea {
                background-color: #2d333b;
                color: #ffffff;
            }
            /* Sidebar button styling */
            [data-testid="stSidebar"] .stButton>button {
                background-color: #444444;
                color: #ffffff;
                border: none;
            }
            /* Header styling */
            h1, h2, h3, h4, h5, h6 {
                color: #ffffff !important;
            }
            /* Custom center title styling */
            .center-title {
                color: white !important;
                font-size: 2.5em;
                text-align: center;
                padding: 1rem 0;
                background-color: #22272e;
                border-radius: 8px;
                margin: 1rem 0;
            }
            /* Override any default header colors */
            .stMarkdown {
                color: #ffffff;
            }
            .tilted-emoji {
                display: inline-block;
                transform: rotate(45deg);
            }
                .tag {
                background-color: #c72c41;
                color: #ffffff;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                margin-right: 0.5rem;
                margin-bottom: 0.5rem;
                display: inline;
                cursor: pointer;
            }
            
            /* Input field with tags */
            .field-input {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                padding: 0.5rem;
                border: 1px solid #444444;
                border-radius: 4px;
            }
            
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("# Web Scraper Settings")
        st.markdown("### Select Model")
        model = st.selectbox(
            "",
            ["gemini-1.5-flash","chatgpt-4o"],
            label_visibility="collapsed"
        )
        st.markdown("### Enter URL")
        url = st.text_input("", label_visibility="collapsed")
        
        fields= st_tags(
            label="Enter Fields to Extract:",
            text="Press enter to add more fields",
            value="",
            maxtags=20,
            key="fields_input"
        )
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
        st.markdown('<div class="button-container"></div>', unsafe_allow_html=True)
        if st.button("Scrape"):
            if not url:
                st.warning("Please enter a URL.")
            else:
                st.success("Scraping started!")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center">
            <h1 style="color: white; font-size: 2.5em;">Universal Web Scraper 🦑</h1>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()