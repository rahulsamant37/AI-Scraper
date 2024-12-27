import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import io

st.set_page_config(page_title="Library Scraper", page_icon="📚", layout="wide")
st.markdown("""
<style>
    .stApp {
        background-color: #2B2B2B;
        color: #FFFFFF;
    }
    .stSelectbox, .stButton>button {
        background-color: #3D3D3D;
        color: #FFFFFF;
        border-radius: 5px;
    }
    .stDataFrame {
        background-color: #3D3D3D;
    }
    .row-widget.stButton {
        text-align: center;
    }
    .download-buttons {
        display: flex;
        justify-content: space-around;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .download-buttons .stButton>button {
        width: 200px;
        color: #FFFFFF;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    .download-buttons .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            120deg,
            transparent,
            rgba(255, 255, 255, 0.2),
            transparent
        );
        transition: all 0.5s;
    }
    .download-buttons .stButton>button:hover::before {
        left: 100%;
    }
    .download-buttons .stButton>button[kind="primary"] {
        background-color: #4CAF50;
        box-shadow: 0 2px 5px rgba(76, 175, 80, 0.3);
    }
    .download-buttons .stButton>button[kind="primary"]:hover {
        background-color: #FF5722;
        box-shadow: 0 5px 15px rgba(255, 87, 34, 0.5);
        transform: translateY(-3px);
    }
    .download-buttons .stButton>button[kind="secondary"] {
        background-color: #008CBA;
        box-shadow: 0 2px 5px rgba(0, 140, 186, 0.3);
    }
    .download-buttons .stButton>button[kind="secondary"]:hover {
        background-color: #FFC107;
        box-shadow: 0 5px 15px rgba(255, 193, 7, 0.5);
        transform: translateY(-3px);
    }
    .download-buttons .stButton>button[kind="tertiary"] {
        background-color: #f44336;
        box-shadow: 0 2px 5px rgba(244, 67, 54, 0.3);
    }
    .download-buttons .stButton>button[kind="tertiary"]:hover {
        background-color: #9C27B0;
        box-shadow: 0 5px 15px rgba(156, 39, 176, 0.5);
        transform: translateY(-3px);
    }
    .stAlert {
        background-color: #4D4D4D;
        color: #FFFFFF;
        border: 1px solid #5A5A5A;
    }
    h1, h2, h3 {
        color: #E0E0E0;
    }
    p {
        color: #CCCCCC;
    }
    .stDataFrame [data-testid="stTable"] {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

def scrape_libraries_and_display(base_url):
    st.title("📚 Library Scraper")
    if 'current_state' not in st.session_state:
        st.session_state.current_state = None
    if 'df' not in st.session_state:
        st.session_state.df = None

    state_options = ["Select a State"] + [link.text.strip() for link in BeautifulSoup(requests.get(base_url).content, "html.parser").find("div", class_="dropdown-content").find_all("a")]
    selected_state = st.selectbox("Select State", state_options, key='state_selector')

    if selected_state != st.session_state.current_state:
        st.session_state.current_state = selected_state
        st.session_state.df = None 

    if st.button("🔍 Scrape Libraries") or (st.session_state.df is not None and selected_state != "Select a State"):
        if selected_state != "Select a State":
            if st.session_state.df is None:
                with st.spinner(f"Scraping data for {selected_state}..."):
                    state_url = base_url + selected_state.lower().replace(" ", "-")
                    state_response = requests.get(state_url)
                    state_soup = BeautifulSoup(state_response.content, "html.parser")
                    libraries_table = state_soup.find("table", id="libraries")
                    if libraries_table is None:
                        st.error(f"No table found for {selected_state}")
                        return
                    headers = [th.text.strip() for th in libraries_table.find_all("th")]
                    rows = []
                    for tr in libraries_table.find_all("tr")[1:]:
                        columns = [td.text.strip() for td in tr.find_all("td")]
                        if columns:
                            rows.append(columns)
                    st.session_state.df = pd.DataFrame(rows, columns=headers)
                st.success(f"Data scraped successfully for {selected_state}!")
            st.markdown("<div class='download-buttons'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="📄 Download as CSV",
                    data=st.session_state.df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{selected_state.lower().replace(' ', '_')}_libraries.csv",
                    mime="text/csv",
                    key="csv_download",
                )
            with col2:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    st.session_state.df.to_excel(writer, index=False, sheet_name='Libraries')
                st.download_button(
                    label="📊 Download as Excel",
                    data=output.getvalue(),
                    file_name=f"{selected_state.lower().replace(' ', '_')}_libraries.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="excel_download",
                )
            
            with col3:
                st.download_button(
                    label="🔢 Download as JSON",
                    data=st.session_state.df.to_json(orient="records"),
                    file_name=f"{selected_state.lower().replace(' ', '_')}_libraries.json",
                    mime="application/json",
                    key="json_download",
                )
            st.markdown("</div>", unsafe_allow_html=True)
            st.subheader(f"Libraries in {selected_state}")
            st.dataframe(st.session_state.df, use_container_width=True)
        else:
            st.warning("Please select a state to scrape libraries.")

def main():
    base_url = "https://publiclibraries.com/state/"
    scrape_libraries_and_display(base_url)

if __name__ == "__main__":
    main()