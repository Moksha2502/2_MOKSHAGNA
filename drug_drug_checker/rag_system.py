"""
RAG (Retrieval-Augmented Generation) system for explaining drug interaction mechanisms.
Uses LangChain with OpenAI for embeddings and LLM generation.
"""

import os
from typing import List, Dict, Optional

# LangChain imports - handle both old and new API
# Use lazy imports to avoid PyTorch DLL errors on Windows
OPENAI_AVAILABLE = False
FAISS = None
OpenAIEmbeddings = None
ChatOpenAI = None
Document = None
ChatPromptTemplate = None
HuggingFaceEmbeddings = None

def _lazy_import_langchain():
    """Lazy import of LangChain modules to avoid DLL errors."""
    global OPENAI_AVAILABLE, FAISS, OpenAIEmbeddings, ChatOpenAI, Document, ChatPromptTemplate, HuggingFaceEmbeddings
    
    if OPENAI_AVAILABLE is not False or FAISS is not None:
        return  # Already imported
    
    dll_error_occurred = False
    
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        OPENAI_AVAILABLE = True
    except OSError as e:
        # Handle DLL initialization failures (Windows error 1114)
        if hasattr(e, 'winerror') and e.winerror == 1114:
            dll_error_occurred = True
            print("\n" + "="*70)
            print("ERROR: PyTorch DLL initialization failed (Windows Error 1114)")
            print("="*70)
            print("This error occurs when PyTorch cannot load its required DLL files.")
            print("\nPossible solutions:")
            print("1. Install Visual C++ Redistributables:")
            print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("2. Reinstall PyTorch:")
            print("   pip uninstall torch torchvision torchaudio")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            print("3. If using CUDA, ensure CUDA drivers are properly installed")
            print("="*70 + "\n")
        else:
            # Other OSError, re-raise it
            raise
    except ImportError as e:
        # Regular import error, try fallbacks
        pass
    except Exception as e:
        # Any other exception
        print(f"Warning: Unexpected error importing langchain modules: {e}")
    
    # If DLL error occurred, don't try fallbacks that also use PyTorch
    if dll_error_occurred:
        FAISS = None
        HuggingFaceEmbeddings = None
        Document = None
        OPENAI_AVAILABLE = False
        return
    
    # Try fallback to older langchain API
    if not OPENAI_AVAILABLE:
        try:
            from langchain.vectorstores import FAISS
            from langchain.embeddings import OpenAIEmbeddings
            from langchain.llms import OpenAI
            from langchain.docstore.document import Document
            ChatOpenAI = OpenAI  # Alias for compatibility
            ChatPromptTemplate = None  # Will use string templates
            OPENAI_AVAILABLE = True
        except (ImportError, OSError):
            # Final fallback to local embeddings (may also fail with DLL error)
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_core.documents import Document
                OPENAI_AVAILABLE = False
            except (ImportError, OSError) as e:
                if isinstance(e, OSError) and hasattr(e, 'winerror') and e.winerror == 1114:
                    print("Warning: HuggingFace embeddings also require PyTorch and failed to load.")
                FAISS = None
                HuggingFaceEmbeddings = None
                Document = None
                OPENAI_AVAILABLE = False

# Call lazy import at module level so OPENAI_AVAILABLE is set correctly when imported
_lazy_import_langchain()


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
        self.use_openrouter = use_openrouter
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
        
        # Lazy import LangChain modules first
        _lazy_import_langchain()
        self.use_openai = use_openai and OPENAI_AVAILABLE
        
        self._initialize_embeddings()
        self._initialize_llm()
        self._build_knowledge_base()
    
    def _initialize_embeddings(self):
        """Initialize embedding model."""
        # Lazy import LangChain modules
        _lazy_import_langchain()
        
        # OpenRouter doesn't support embeddings API, so use local embeddings when using OpenRouter
        if self.use_openrouter:
            print("Note: Using local embeddings (OpenRouter doesn't support embeddings API)")
            try:
                if HuggingFaceEmbeddings:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    print("Local embeddings initialized successfully.")
                else:
                    print("Warning: HuggingFace embeddings not available. RAG features will be limited.")
                    self.embeddings = None
            except Exception as e:
                print(f"Warning: Could not load local embeddings model: {e}")
                print("The application will work in template-based mode without RAG features.")
                self.embeddings = None
        elif self.use_openai and self.api_key:
            # Only use OpenAI embeddings when NOT using OpenRouter
            try:
                # Use OpenAI embeddings
                if OpenAIEmbeddings:
                    # Test the API key by trying to create embeddings (this will fail fast if invalid)
                    self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
                    # Try a simple test to validate the key
                    try:
                        # This will fail immediately if the key is invalid
                        test_embedding = self.embeddings.embed_query("test")
                    except Exception as key_error:
                        # Check if it's an authentication error
                        error_str = str(key_error).lower()
                        if "401" in error_str or "invalid" in error_str or "api key" in error_str or "unauthorized" in error_str:
                            print(f"Warning: Invalid API key provided. Error: {key_error}")
                            print("The application will work in template-based mode without embeddings.")
                            self.embeddings = None
                            self.use_openai = False
                        else:
                            raise
                else:
                    raise ImportError("OpenAIEmbeddings not available")
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's an authentication error
                if "401" in error_str or "invalid" in error_str or "api key" in error_str or "unauthorized" in error_str:
                    print(f"Warning: Invalid or expired API key. Error: {e}")
                    print("The application will work in template-based mode without AI features.")
                else:
                    print(f"Warning: Could not initialize OpenAI embeddings: {e}")
                self.embeddings = None
                self.use_openai = False
        
        # Fallback to local embeddings if OpenAI not available and not using OpenRouter
        if not self.embeddings and not self.use_openrouter:
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
        # Lazy import LangChain modules
        _lazy_import_langchain()
        
        if self.use_openai and self.api_key:
            try:
                if not ChatOpenAI:
                    raise ImportError("ChatOpenAI not available")
                    
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
                error_str = str(e).lower()
                # Check if it's an authentication error
                if "401" in error_str or "invalid" in error_str or "api key" in error_str or "unauthorized" in error_str:
                    print(f"Warning: Invalid or expired API key. Error: {e}")
                    print("The application will work in template-based mode without LLM features.")
                else:
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
                error_str = str(e).lower()
                # Check if it's an authentication error
                if "401" in error_str or "invalid" in error_str or "api key" in error_str or "unauthorized" in error_str:
                    print(f"Warning: Invalid API key - could not build vector store. Error: {e}")
                    print("The application will work in template-based mode without RAG features.")
                    self.embeddings = None
                    self.use_openai = False
                else:
                    print(f"Warning: Could not build vector store: {e}")
                    print("The application will work in template-based mode.")
    
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
        
        # Use LLM if available (even without context_text for better explanations)
        if self.llm:
            try:
                if context_text:
                    print(f"[DEBUG] RAG: Using LLM with context (context_text length: {len(context_text)})")
                else:
                    print(f"[DEBUG] RAG: Using LLM without context (will use provided information only)")
                # Lazy import ChatPromptTemplate
                _lazy_import_langchain()
                
                if ChatPromptTemplate:
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
{context_section}
Please provide a comprehensive explanation of this drug interaction, including:
1. What the interaction means
2. Why it occurs (mechanism) - provide detailed medical explanation
3. Potential risks and clinical significance
4. Safety recommendations based on the severity level
5. Any relevant clinical considerations

Be professional, clear, detailed, and emphasize consulting healthcare professionals. Provide a thorough medical explanation."""),
                    ])
                    
                    chain = prompt | self.llm
                    context_section = f"\nRelevant Context from Database:\n{context_text}\n" if context_text else "\n(No additional context available from database)\n"
                    response = chain.invoke({
                        "drug1": drug1,
                        "drug2": drug2,
                        "severity": severity,
                        "mechanism": mechanism or "Not specified",
                        "description": description or "No detailed description available",
                        "context_section": context_section
                    })
                    
                    return response.content if hasattr(response, 'content') else str(response)
                else:
                    # Fallback: use LLM directly with string formatting
                    context_section = f"\nRelevant Context from Database:\n{context_text}\n" if context_text else "\n(No additional context available from database)\n"
                    prompt_text = f"""You are a medical assistant explaining drug-drug interactions.

Drug Interaction Information:
Drug 1: {drug1}
Drug 2: {drug2}
Severity: {severity}
Mechanism: {mechanism or "Not specified"}
Description: {description or "No detailed description available"}
{context_section}
Please provide a comprehensive, detailed explanation of this drug interaction, including:
1. What the interaction means clinically
2. Why it occurs (detailed mechanism explanation)
3. Potential risks and clinical significance
4. Safety recommendations based on severity
5. Any relevant clinical considerations

Be professional, clear, detailed, and emphasize consulting healthcare professionals."""
                    
                    response = self.llm.invoke(prompt_text)
                    return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                print(f"[WARNING] RAG LLM generation failed, using template: {e}")
                # Fall through to template-based generation
        else:
            if not self.llm:
                print(f"[DEBUG] RAG: No LLM available (llm={self.llm})")
            if not context_text:
                print(f"[DEBUG] RAG: No context_text available (context_text length: {len(context_text) if context_text else 0})")
        
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

