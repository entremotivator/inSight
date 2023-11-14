import streamlit as st


def main():
    st.set_page_config(page_title="Research InSight", page_icon=":books:")
    st.header("Dive deeper into your researches :books:")
    st.text_input("Ask a question about your document")

    with st.sidebar:
        st.subheader("Your Research PDFs")
        st.file_uploader("Upload your PDFs here!")
        st.button("Process")

if __name__ == "__main__":
    main()