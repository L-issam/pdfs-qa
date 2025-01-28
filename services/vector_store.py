import logging
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from services.embeddings_manager import EmbeddingsManager
from utils.hardware_manager import HardwareManager
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from langchain.chains import RetrievalQA
from pathlib import Path

class VectorStore:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.hardware_manager = HardwareManager()
        self.embeddings_manager = EmbeddingsManager()
        
        # Embeddings pour la recherche
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Remplacer LlamaCpp par T5
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        
        # Optimisations
        self.model.eval()  # Mode inférence
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        self.vector_store = None
        
        # Ajouter un classifieur de domaine
        self.domains = {
            'technique': ['norme', 'métier', 'technique', 'maille', 'couture', 'siège', 'automobile', 'voiture électrique', 'pièces auto', 'fournisseurs automobile'],
            'administratif': ['préfecture', 'email', 'adresse', 'formulaire', 'admission', 'dossier', 'documents', 'justificatifs', 'rendez-vous', 'séjour', 'carte de séjour', 'carte sejour'],
            'digital': ['site web', 'référencement', 'SEO', 'médias sociaux', 'publicité en ligne', 'emailing', 'contenu', 'branding', 'influenceurs', 'analytics', 'e-commerce'],  
            'juridique': ['contrat', 'loi', 'réglementation', 'tribunal', 'avocat', 'litige', 'contentieux', 'droit civil', 'jurisprudence', 'procédure', 'code pénal'],  
            'finance': ['budget', 'investissement', 'bilan', 'comptabilité', 'fiscalité', 'audit', 'trésorerie', 'placement', 'banque', 'prêt', 'épargne'],  
            'immobilier': ['achat', 'vente', 'location', 'bail', 'notaire', 'hypothèque', 'diagnostic', 'agence', 'propriétaire', 'locataire', 'cadastre'],  
            'santé': ['consultation', 'ordonnance', 'traitement', 'assurance', 'mutuelle', 'hospitalisation', 'prévention', 'médicament', 'diagnostic', 'chirurgie', 'urgence'],  
            'logistique': ['transport', 'chaîne d\'approvisionnement', 'entreposage', 'expédition', 'livraison', 'stockage', 'fret', 'inventaire', 'gestion des stocks', 'emballage', 'supply chain'],  
            'éducation': ['école', 'université', 'formation', 'cours', 'diplôme', 'enseignant', 'apprentissage', 'pédagogie', 'examen', 'programme', 'inscription'],  
            'informatique': ['logiciel', 'matériel', 'programmation', 'cybersécurité', 'réseau', 'base de données', 'serveur', 'cloud', 'développement', 'intelligence artificielle', 'algorithmique'],  
            'alimentaire': ['nutrition', 'ingrédients', 'recette', 'cuisine', 'restauration', 'production', 'distribution', 'hygiène', 'sécurité alimentaire', 'conservation', 'emballage'],  
            'tourisme': ['voyage', 'destination', 'hébergement', 'réservation', 'tour opérateur', 'guide', 'avion', 'visa', 'culture', 'hôtellerie', 'forfait']
        }
        
        # Créer un store par domaine
        self.vector_stores = {}

    def _initialize_faiss_index(self, dimension: int) -> faiss.Index:
        """Initialize FAISS index with the appropriate device."""
        try:
            if self.hardware_manager.device == 'cuda':
                self.logger.info("Initializing FAISS index on GPU")
                res = faiss.StandardGpuResources()
                index = faiss.IndexFlatL2(dimension)
                return faiss.index_cpu_to_gpu(res, 0, index)
            else:
                self.logger.info("Initializing FAISS index on CPU")
                return faiss.IndexFlatL2(dimension)
        except Exception as e:
            self.logger.error(f"Error initializing FAISS index: {str(e)}")
            self.logger.info("Falling back to CPU index")
            return faiss.IndexFlatL2(dimension)

    def add_documents(self, documents: List[Document]) -> int:
        """Ajoute les documents en les séparant par domaine."""
        try:
            for doc in documents:
                # Classifier le document
                domain = self._classify_document(doc)
                
                # Créer ou obtenir le store du domaine
                if domain not in self.vector_stores:
                    self.vector_stores[domain] = FAISS.from_documents([doc], self.embeddings)
                else:
                    self.vector_stores[domain].add_documents([doc])
                
            return len(documents)
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            return 0

    def _classify_document(self, doc: Document) -> str:
        """Classifie un document selon son contenu."""
        content = doc.page_content.lower()
        
        # Compter les mots-clés de chaque domaine
        domain_scores = {}
        for domain, keywords in self.domains.items():
            score = sum(1 for keyword in keywords if keyword in content)
            domain_scores[domain] = score
        
        # Retourner le domaine avec le plus de correspondances
        return max(domain_scores.items(), key=lambda x: x[1])[0]

    def search(self, query: str) -> List[Dict]:
        try:
            # Détecter le domaine de la question
            query_lower = query.lower()
            query_domain = None
            for domain, keywords in self.domains.items():
                if any(keyword in query_lower for keyword in keywords):
                    query_domain = domain
                    break
            
            # Si domaine identifié, chercher d'abord dans ce domaine
            if query_domain and query_domain in self.vector_stores:
                source_docs = self.vector_stores[query_domain].similarity_search_with_score(query, k=2)
                if source_docs:
                    return self._process_search_results(source_docs, query)
            
            # Sinon, chercher dans tous les domaines
            all_results = []
            for domain, store in self.vector_stores.items():
                results = store.similarity_search_with_score(query, k=1)
                all_results.extend(results)
            
            # Trier par score
            all_results.sort(key=lambda x: x[1])
            return self._process_search_results(all_results[:2], query)
            
        except Exception as e:
            self.logger.error(f"Erreur: {str(e)}")
            return [{"content": "Erreur lors de la recherche.", "sources": []}]

    def _generate_summary(self) -> List[Dict]:
        """Génère un résumé des documents."""
        try:
            # Récupérer tous les documents
            all_docs = self.vector_store.similarity_search("", k=100)
            
            # Grouper par source
            docs_by_source = {}
            for doc in all_docs:
                source = doc.metadata["source"]
                if source not in docs_by_source:
                    docs_by_source[source] = []
                docs_by_source[source].append(doc.page_content)

            # Créer un résumé par document
            summaries = []
            for source, contents in docs_by_source.items():
                summaries.append(f"• {source}: {contents[0][:200]}...")

            return [{
                "content": "Documents disponibles :\n" + "\n".join(summaries),
                "sources": [{
                    "content": "\n".join(contents),
                    "source": source,
                    "relevance": "100%"
                } for source, contents in docs_by_source.items()]
            }]

        except Exception as e:
            self.logger.error(f"Erreur lors du résumé: {str(e)}")
            return [{"content": "Erreur lors de la génération du résumé.", "sources": []}]

    def _extract_relevant_answer(self, content: str, query: str) -> str:
        """Extrait une réponse pertinente avec une recherche plus souple."""
        # Mots clés importants de la question
        query_keywords = set(query.lower().split())
        important_keywords = {
            'email': ['email', 'mail', 'adresse', 'courriel', '@'],
            'prefecture': ['prefecture', 'préfecture', 'pref', 'séjour', 'sejour', 'carte de séjour', 'carte sejour'],
            # Ajouter d'autres groupes de synonymes selon les besoins
        }
        
        # Enrichir les mots clés de recherche avec les synonymes
        expanded_keywords = set()
        for word in query_keywords:
            for group in important_keywords.values():
                if word in group:
                    expanded_keywords.update(group)
        query_keywords.update(expanded_keywords)
        
        # Diviser le contenu en phrases
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return "Information non trouvée."
        
        # Fonction de score pour évaluer la pertinence d'une phrase
        def score_sentence(sentence):
            sentence_lower = sentence.lower()
            # Vérifier la présence d'une adresse email
            if '@' in sentence and any(k in query_keywords for k in ['email', 'mail', 'adresse']):
                return 100  # Score élevé pour les phrases contenant un email si demandé
            # Score basé sur le nombre de mots clés trouvés
            return sum(1 for word in query_keywords if word in sentence_lower)
        
        # Trouver les phrases les plus pertinentes
        scored_sentences = [(score_sentence(s), s) for s in sentences]
        relevant_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)
        
        # Retourner les phrases pertinentes (score > 0)
        relevant_responses = [sent for score, sent in relevant_sentences if score > 0]
        
        if not relevant_responses:
            return "Information non trouvée."
        
        # Retourner la ou les phrases pertinentes
        return ". ".join(relevant_responses[:2]) + "."

    def _format_content(self, content: str, query: str) -> str:
        """Formate le contenu pour une meilleure lisibilité."""
        # Extraire la partie la plus pertinente
        sentences = content.split('.')
        formatted_content = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                formatted_content.append(f"• {sentence}.")
                
        return "\n".join(formatted_content)

    def _format_result(self, doc: Any, score: float) -> Dict[str, Any]:
        """Format search results."""
        confidence = max(0, min(1, 1 - (score / 100)))
        return {
            'context': doc.page_content,
            'source': doc.metadata.get('source', 'Unknown'),
            'page': doc.metadata.get('page', 0) + 1,
            'confidence': confidence
        }

    def _generate_with_model(self, prompt: str) -> str:
        # Formater le prompt pour T5
        formatted_prompt = f"answer: {prompt}"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", max_length=512, truncation=True)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        outputs = self.model.generate(
            **inputs,
            max_length=100,        # Réduire la longueur max
            min_length=20,         # Réduire la longueur min
            temperature=0.3,       # Réduire la température pour plus de cohérence
            top_p=0.9,            # Filtrer les tokens moins probables
            repetition_penalty=1.2,# Pénaliser les répétitions
            num_return_sequences=1
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Si la réponse est hors sujet ou répétitive
        if "will respond" in response.lower() or len(response.split()) < 5:
            return "Je ne peux pas répondre à cette question car elle ne concerne pas le contenu des documents."
        
        return response

    def _process_search_results(self, results: List[Tuple[Document, float]], query: str) -> List[Dict]:
        """Process search results and return formatted responses."""
        if not results:
            self.logger.warning("Aucun résultat trouvé dans l'index")
            return [{"content": "Pas d'information trouvée dans les documents.", "sources": []}]

        # Vérifier si la question est liée aux documents
        source_docs = results
        best_doc, score = source_docs[0]
        
        # Log pour debug
        self.logger.info(f"Score de similarité: {score}")
        self.logger.info(f"Contenu trouvé: {best_doc.page_content[:200]}...")
        
        # Extraire la réponse
        answer = self._extract_relevant_answer(best_doc.page_content, query)
        self.logger.info(f"Réponse extraite: {answer}")
        
        # Retourner tous les résultats pertinents
        return [{
            "content": answer,
            "sources": [{
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "page": int(doc.metadata.get("page", 0)) + 1,
                "relevance": f"{(1 - score) * 100:.1f}%"
            } for doc, score in source_docs]
        }]

    def summarize_pdf(self, pdf_path: Path) -> str:
        """Génère un résumé intelligent du PDF."""
        try:
            # Récupérer les chunks du PDF
            results = []
            for store in self.vector_stores.values():
                docs = store.similarity_search(
                    f"contenu principal de {pdf_path.name}",
                    k=10,
                    filter={"source": pdf_path.name}
                )
                results.extend(docs)
            
            if not results:
                return "Impossible de générer un résumé pour ce document."
            
            # Trier par page et compter les pages
            results.sort(key=lambda x: x.metadata.get('page', 0))
            num_pages = max(doc.metadata.get('page', 0) for doc in results)
            
            # Préparer le contexte
            context = "\n".join([doc.page_content for doc in results])
            
            # Prompt caché du résultat final
            inputs = self.tokenizer.encode(
                f"Résume ce document en expliquant son type, son objectif et ses points clés : {context[:2000]}",
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # Paramètres de génération corrigés
            outputs = self.model.generate(
                inputs,
                max_length=500,
                min_length=100,
                do_sample=True,  # Activer l'échantillonnage
                temperature=0.7,
                num_beams=4,
                no_repeat_ngram_size=3
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Formater sans montrer le prompt
            return f"""# Résumé de {pdf_path.name}

{summary}

Document de {num_pages} pages"""
            
        except Exception as e:
            self.logger.error(f"Erreur lors du résumé: {str(e)}")
            return "Erreur lors de la génération du résumé."
