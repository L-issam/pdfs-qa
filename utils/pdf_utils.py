import streamlit as st
from pathlib import Path

def delete_pdf(pdf_path: Path, processed_pdfs: set):
    """Fonction centralisée pour supprimer un PDF."""
    try:
        # Supprimer le fichier
        pdf_path.unlink()
        # Supprimer du vector store
        if 'vector_store' in st.session_state:
            st.session_state.vector_store = None  # Forcer la réindexation
        # Supprimer des PDFs traités
        processed_pdfs.remove(pdf_path.name)
        st.success(f"{pdf_path.name} supprimé")
        st.rerun()
    except Exception as e:
        st.error(f"Erreur lors de la suppression de {pdf_path.name}: {str(e)}") 