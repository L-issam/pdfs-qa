import streamlit as st
from pathlib import Path
from typing import Optional
from streamlit_pdf_viewer import pdf_viewer
import io
from PyPDF2 import PdfReader, PdfWriter

class PDFViewer:
    @staticmethod
    def display_pdf(pdf_path: Optional[Path], page: int = 1, highlight_text: str = None) -> None:
        if pdf_path is None:
            st.info("Sélectionnez un PDF dans la barre latérale.")
            return

        try:
            # Créer un PDF avec uniquement la page demandée
            pdf_reader = PdfReader(pdf_path)
            pdf_writer = PdfWriter()
            
            # Page numbers are 0-based in PyPDF2
            page_idx = page - 1
            if 0 <= page_idx < len(pdf_reader.pages):
                pdf_writer.add_page(pdf_reader.pages[page_idx])
                
                # Sauvegarder en mémoire
                output_pdf = io.BytesIO()
                pdf_writer.write(output_pdf)
                output_pdf.seek(0)
                
                # Afficher le PDF
                pdf_viewer(output_pdf.read())
                
                # Ajouter un bouton pour voir le PDF complet
                st.markdown("---")  # Séparateur
                if st.button("📄 Voir le document entier"):
                    with open(pdf_path, "rb") as f:
                        pdf_viewer(f.read())
            else:
                st.error(f"Page {page} not found in document")
            
        except Exception as e:
            st.error(f"Erreur d'affichage du PDF: {str(e)}")