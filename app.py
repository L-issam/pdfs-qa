import streamlit as st
from pathlib import Path
from components.pdf_viewer import PDFViewer
from components.sidebar import Sidebar
from components.qa_interface import QAInterface
from services.vector_store import VectorStore
from services.pdf_processor import PDFProcessor
from typing import List, Dict
from utils.pdf_utils import delete_pdf  # Import depuis utils

def main():
    st.set_page_config(page_title="Recherche dans les PDFs", layout="wide")

    # Initialize session state
    if 'processed_pdfs' not in st.session_state:
        st.session_state.processed_pdfs = set()
        
    # Réinitialiser le vector store s'il est None
    if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
        st.session_state.vector_store = VectorStore()
        
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
        
    # Charger les PDFs existants au démarrage
    documents = st.session_state.pdf_processor.process_file()
    if documents:
        st.session_state.vector_store.add_documents(documents)
        for doc in documents:
            st.session_state.processed_pdfs.add(doc.metadata["source"])

    if 'current_pdf' not in st.session_state:
        st.session_state.current_pdf = None

    # Initialize components
    sidebar = Sidebar(st.session_state.pdf_processor.sources_dir)
    qa_interface = QAInterface()
    pdf_viewer = PDFViewer()

    # Create two columns for main layout
    main_col, qa_col = st.columns([2, 1])

    with main_col:
        st.title("📚 Recherche dans un ensemble de PDFs")
        
        # Section upload
        uploaded_files = st.file_uploader(
            "Déposez vos documents PDF",
            type=['pdf'],
            accept_multiple_files=True
        )

        # Afficher les fichiers uploadés avec bouton de suppression
        for file in uploaded_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                if st.button("🗑️", key=f"main_delete_{file.name}"):
                    pdf_path = Path("sources") / file.name
                    delete_pdf(pdf_path, st.session_state.processed_pdfs)

        # Messages de traitement
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.processed_pdfs:
                    with st.spinner(f'Traitement de {file.name}...'):
                        try:
                            documents = st.session_state.pdf_processor.process_file(file)
                            st.session_state.vector_store.add_documents(documents)
                            st.session_state.processed_pdfs.add(file.name)
                            st.success(f"{file.name} traité avec succès")
                        except Exception as e:
                            st.error(f"Erreur lors du traitement de {file.name}: {str(e)}")

        # Affichage du PDF sélectionné
        if st.session_state.current_pdf:
            page = getattr(st.session_state, 'current_page', 1)
            highlight = getattr(st.session_state, 'highlight_text', None)
            pdf_viewer.display_pdf(
                st.session_state.current_pdf,
                page=page,
                highlight_text=highlight
            )
        else:
            st.info("Sélectionnez un PDF dans la barre latérale pour le visualiser.")

    with qa_col:
        # Question and Answer interface
        query = qa_interface.render_question_input()
        if query:
            with st.spinner("Searching for answers..."):
                answers = st.session_state.vector_store.search(query)
                qa_interface.render_answers(answers)

    # Render sidebar and handle PDF selection
    selected_pdf_path = sidebar.render(st.session_state.processed_pdfs)
    if selected_pdf_path:
        st.session_state.current_pdf = selected_pdf_path

    # Gérer la demande de résumé
    if hasattr(st.session_state, 'summarize_pdf') and st.session_state.summarize_pdf:
        pdf_path = st.session_state.summarize_pdf
        with st.spinner(f"Génération du résumé de {pdf_path.name}..."):
            summary = st.session_state.vector_store.summarize_pdf(pdf_path)
            st.markdown("### 📝 Résumé du document")
            st.markdown(summary)
            # Réinitialiser pour ne pas regénérer à chaque refresh
            st.session_state.summarize_pdf = None

if __name__ == "__main__":
    main()
