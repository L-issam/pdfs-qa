import logging
from pathlib import Path
from typing import List
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sources_dir = Path("sources")
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def process_file(self, uploaded_file=None):
        """Traite un fichier PDF et retourne une liste de Documents."""
        documents = []

        try:
            # Si un fichier est uploadé, le traiter
            if uploaded_file:
                save_path = self.sources_dir / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                doc = self._load_pdf(save_path)
                if doc:  # Vérifier que le document n'est pas vide
                    documents.extend(doc)

            # Si pas de fichier uploadé, traiter les fichiers existants
            else:
                for pdf_path in self.sources_dir.glob("*.pdf"):
                    if pdf_path.stat().st_size > 0:  # Vérifier que le fichier n'est pas vide
                        doc = self._load_pdf(pdf_path)
                        if doc:  # Vérifier que le document a été chargé correctement
                            documents.extend(doc)

            # Ne retourner les documents que s'il y en a
            if documents:
                self.logger.info(f"Successfully processed {len(documents)} document chunks")
                return documents
            else:
                self.logger.warning("No valid documents to process")
                return []
                
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            return []  # Retourner une liste vide en cas d'erreur

    def _load_pdf(self, pdf_path: Path) -> List[Document]:
        """Charge et traite un fichier PDF."""
        try:
            documents = []
            pdf_reader = PdfReader(str(pdf_path))
            
            # Traiter chaque page séparément
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:  # Vérifier que le texte n'est pas vide
                    # Nettoyer le texte de la page
                    cleaned_text = self._clean_text(page_text)
                    
                    # Créer des chunks pour cette page
                    page_docs = self.text_splitter.create_documents(
                        texts=[cleaned_text],
                        metadatas=[{
                            "source": pdf_path.name,
                            "page": page_num  # Garder le numéro de page pour chaque chunk
                        }]
                    )
                    documents.extend(page_docs)

            if not documents:  # Vérifier qu'on a des documents
                self.logger.warning(f"No text content found in {pdf_path.name}")
                return []

            return documents

        except Exception as e:
            self.logger.error(f"Error loading PDF {pdf_path.name}: {str(e)}")
            return []

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte."""
        text = " ".join(text.split())
        text = text.replace('\n', ' ').replace('\r', ' ')
        return ' '.join(text.split())

    def get_processed_files(self) -> List[Path]:
        """Retourne la liste des fichiers PDF dans le dossier sources."""
        return list(self.sources_dir.glob("*.pdf"))