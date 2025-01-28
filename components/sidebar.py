import streamlit as st
from pathlib import Path
from typing import Optional
from utils.pdf_utils import delete_pdf  # Import depuis utils

class Sidebar:
    def __init__(self, sources_dir: Path):
        self.sources_dir = sources_dir

    def render(self, processed_pdfs: set) -> Optional[Path]:
        with st.sidebar:
            st.markdown("📑 Documents Traités")
            st.markdown("### PDFs Disponibles")
            
            # Utiliser un container pour éviter le rechargement
            container = st.container()
            
            # Stocker la sélection dans session_state
            if 'selected_pdf' not in st.session_state:
                st.session_state.selected_pdf = None
            
            for pdf in sorted(processed_pdfs):
                pdf_path = self.sources_dir / pdf
                col1, col2 = container.columns([4, 1])
                
                # Bouton pour voir le PDF
                if col1.button(
                    f"📄 {pdf}",
                    key=f"btn_{pdf}",
                    help="Cliquer pour voir ce PDF",
                    use_container_width=True,
                    type="secondary"
                ):
                    st.session_state.selected_pdf = pdf_path
                
                # Bouton pour supprimer
                if col2.button("🗑️", key=f"del_{pdf}", help="Supprimer ce PDF"):
                    delete_pdf(pdf_path, processed_pdfs)
            
            return st.session_state.selected_pdf
