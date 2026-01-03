"""
RAG (Retrieval-Augmented Generation) system for explaining drug interaction mechanisms.
Uses LangChain with OpenAI for embeddings and LLM generation.
"""

import os
from typing import List, Dict, Optional

# LangChain imports - handle both old and new API
try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    OPENAI_AVAILABLE = True
except ImportError:
    try:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.llms import OpenAI
        from langchain.docstore.document import Document
        OPENAI_AVAILABLE = True
    except ImportError:
        # Fallback to local embeddings if OpenAI not available
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_core.documents import Document
            OPENAI_AVAILABLE = False
        except ImportError:
            FAISS = None
            HuggingFaceEmbeddings = None
            Document = None
            OPENAI_AVAILABLE = False


class RAGSystem:
    """RAG system for generating drug interaction explanations using OpenAI."""
    
    def __init__(self, interactions_data: Optional[List[Dict]] = None, api_key: Optional[str] = None, 
                 use_openai: bool = True, use_openrouter: bool = False, model_name: str = "openai/gpt-3.5-turbo"):
        """
        Initialize RAG system.
        
        Args:
            interactions_data: List of interaction dictionaries to build knowledge base
            api_key: OpenRouter or OpenAI API key. If None, tries to get from environment.
            use_openai: Whether to use OpenAI/OpenRouter (True) or local embeddings (False)
            use_openrouter: Whether to use OpenRouter API (True) or direct OpenAI (False)
            model_name: Model to use for LLM (e.g., "openai/gpt-3.5-turbo" for OpenRouter)
        """
        self.interactions_data = interactions_data or []
        self.vectorstore = None
        self.embeddings = None
        self.retriever = None
        self.llm = None
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.use_openrouter = use_openrouter
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
        
        self._initialize_embeddings()
        self._initialize_llm()
        self._build_knowledge_base()
    
    def _initialize_embeddings(self):
        """Initialize embedding model."""
        if self.use_openai and self.api_key:
            try:
                # Use OpenAI embeddings
                self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI embeddings: {e}")
                self.embeddings = None
                self.use_openai = False
        
        # Fallback to local embeddings if OpenAI not available
        if not self.embeddings:
            try:
                if HuggingFaceEmbeddings:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
            except Exception as e:
                print(f"Warning: Could not load local embeddings model: {e}")
                self.embeddings = None
    
    def _initialize_llm(self):
        """Initialize LLM for generation."""
        if self.use_openai and self.api_key:
            try:
                if self.use_openrouter:
                    # Use OpenRouter API
                    self.llm = ChatOpenAI(
                        model=self.model_name,
                        temperature=0.3,
                        openai_api_key=self.api_key,
                        openai_api_base="https://openrouter.ai/api/v1",
                        default_headers={"HTTP-Referer": "https://github.com/your-repo"}  # Optional: for tracking
                    )
                else:
                    # Use direct OpenAI API
                    model = self.model_name.replace("openai/", "") if "openai/" in self.model_name else self.model_name
                    # Try both model and model_name for compatibility
                    try:
                        self.llm = ChatOpenAI(
                            model=model,
                            temperature=0.3,
                            openai_api_key=self.api_key
                        )
                    except:
                        self.llm = ChatOpenAI(
                            model_name=model,
                            temperature=0.3,
                            openai_api_key=self.api_key
                        )
            except Exception as e:
                print(f"Warning: Could not initialize LLM: {e}")
                self.llm = None
    
    def _build_knowledge_base(self):
        """Build vector store from interactions data."""
        if not self.interactions_data or not self.embeddings or FAISS is None or Document is None:
            return
        
        # Create documents from interactions
        documents = []
        for interaction in self.interactions_data:
            # Create a comprehensive document text
            doc_text = f"""
            Drug Pair: {interaction.get('drug1', '')} and {interaction.get('drug2', '')}
            Interaction Type: {interaction.get('interaction_type', 'Unknown')}
            Severity: {interaction.get('severity', 'Moderate')}
            Mechanism: {interaction.get('mechanism', '')}
            Description: {interaction.get('description', '')}
            """
            documents.append(Document(
                page_content=doc_text.strip(),
                metadata={
                    'drug1': interaction.get('drug1', ''),
                    'drug2': interaction.get('drug2', ''),
                    'severity': interaction.get('severity', 'Moderate'),
                    'mechanism': interaction.get('mechanism', '')
                }
            ))
        
        if documents and self.embeddings:
            try:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
            except Exception as e:
                print(f"Warning: Could not build vector store: {e}")
    
    def retrieve_context(self, drug1: str, drug2: str, mechanism: str = "") -> List[Document]:
        """Retrieve relevant context for a drug pair."""
        if not self.retriever:
            return []
        
        query = f"{drug1} {drug2} interaction {mechanism}"
        try:
            # Handle both old and new LangChain API
            if hasattr(self.retriever, 'get_relevant_documents'):
                docs = self.retriever.get_relevant_documents(query)
            elif hasattr(self.retriever, 'invoke'):
                docs = self.retriever.invoke(query)
            else:
                docs = []
            return docs
        except Exception as e:
            print(f"Warning: Error retrieving context: {e}")
            return []
    
    def generate_explanation(self, drug1: str, drug2: str, mechanism: str, 
                           severity: str, description: str = "") -> str:
        """
        Generate explanation for a drug interaction using RAG with LLM.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            mechanism: Interaction mechanism
            severity: Interaction severity
            description: Interaction description
            
        Returns:
            Generated explanation
        """
        # Retrieve relevant context
        context_docs = self.retrieve_context(drug1, drug2, mechanism)
        context_text = ""
        if context_docs:
            context_text = "\n".join([doc.page_content for doc in context_docs[:3]])
        
        # Use LLM if available, otherwise use template-based generation
        if self.llm and context_text:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a medical assistant explaining drug-drug interactions. 
Provide clear, accurate, and helpful explanations based on the provided context.
Always include safety recommendations based on severity level."""),
                    ("human", """Drug Interaction Information:
Drug 1: {drug1}
Drug 2: {drug2}
Severity: {severity}
Mechanism: {mechanism}
Description: {description}

Relevant Context from Database:
{context}

Please provide a comprehensive explanation of this drug interaction, including:
1. What the interaction means
2. Why it occurs (mechanism)
3. Potential risks
4. Safety recommendations based on the severity level

Be professional, clear, and emphasize consulting healthcare professionals."""),
                ])
                
                chain = prompt | self.llm
                response = chain.invoke({
                    "drug1": drug1,
                    "drug2": drug2,
                    "severity": severity,
                    "mechanism": mechanism or "Not specified",
                    "description": description or "No detailed description available",
                    "context": context_text
                })
                
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                print(f"Warning: LLM generation failed, using template: {e}")
                # Fall through to template-based generation
        
        # Template-based fallback
        explanation_parts = []
        
        if description:
            explanation_parts.append(description)
        elif mechanism:
            explanation_parts.append(f"Mechanism: {mechanism}")
        
        # Add context if available
        if context_text:
            explanation_parts.append(f"\nAdditional context: Similar interaction patterns have been documented.")
        
        # Add safety recommendations based on severity
        severity_lower = severity.lower()
        if 'contraindicated' in severity_lower or 'major' in severity_lower:
            explanation_parts.append("\n[!] RECOMMENDATION: Avoid concurrent use. Consider alternative medications.")
        elif 'moderate' in severity_lower:
            explanation_parts.append("\n[!] RECOMMENDATION: Use with caution. Monitor for adverse effects and consider dose adjustments.")
        else:
            explanation_parts.append("\n[!] RECOMMENDATION: Monitor patient closely when using these medications together.")
        
        explanation = " ".join(explanation_parts)
        
        # If we have a good mechanism, enhance the explanation
        if mechanism and not description:
            explanation = f"This interaction occurs through {mechanism.lower()}. " + explanation
        
        return explanation.strip()
    
    def enhance_interaction(self, interaction: Dict) -> Dict:
        """Enhance interaction dictionary with RAG-generated explanation."""
        explanation = self.generate_explanation(
            drug1=interaction.get('drug1', ''),
            drug2=interaction.get('drug2', ''),
            mechanism=interaction.get('mechanism', ''),
            severity=interaction.get('severity', 'Moderate'),
            description=interaction.get('description', '')
        )
        
        interaction['explanation'] = explanation
        return interaction

