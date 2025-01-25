import streamlit as st
from typing import List, Dict
from pathlib import Path
from components.pdf_viewer import PDFViewer

class QAInterface:
    def render_question_input(self) -> str:
        st.markdown("### 🤔 Trouver dans CES documents")
        return st.text_input("", placeholder="Une question...")

    def render_answers(self, answers: List[Dict]):
        if not answers:
            st.warning("Pas de résultat trouvé")
            return

        answer = answers[0]
        st.markdown("### ✎ Réponse")
        st.markdown(answer["content"])
        
        if answer["sources"]:
            st.markdown("### § Sources")
            for i, source in enumerate(answer["sources"], 1):
                with st.expander(f"Source {i} - Page {source['page']} - {source['source']}"):
                    st.markdown(source["content"])
                    if st.button(f"📄 Voir dans le PDF", key=f"pdf_btn_{i}"):
                        pdf_path = Path("sources") / source['source']
                        st.session_state.current_pdf = pdf_path
                        st.session_state.current_page = source.get('page', 1)
                        st.session_state.highlight_text = source["content"]
                        st.rerun()