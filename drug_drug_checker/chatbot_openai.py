"""
OpenRouter-powered chatbot for drug interaction queries with RAG.
Integrates with the drug interaction checker backend and RAG system.
Supports OpenRouter API which provides access to multiple LLMs.
"""

import os
from typing import List, Dict, Optional
from drug_interaction_checker import DrugInteractionChecker
from rag_system import RAGSystem

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, SystemMessage
    OPENAI_AVAILABLE = True
except ImportError:
    try:
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        print("Warning: langchain-openai not installed. Install with: pip install langchain-openai")


class DrugInteractionChatbot:
    """Chatbot for drug interaction queries using OpenRouter/OpenAI with RAG."""
    
    def __init__(self, api_key: Optional[str] = None, data_path: Optional[str] = None, 
                 use_openrouter: bool = True, model_name: str = "openai/gpt-3.5-turbo"):
        """
        Initialize the chatbot.
        
        Args:
            api_key: OpenRouter or OpenAI API key. If None, tries to get from environment.
            data_path: Path to DDInter CSV file.
            use_openrouter: Whether to use OpenRouter API (True) or direct OpenAI (False).
            model_name: Model to use. For OpenRouter, use format like "openai/gpt-3.5-turbo" or "anthropic/claude-3-haiku"
        """
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            # Try OpenRouter first, then OpenAI
            self.api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("API key not provided. Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.use_openrouter = use_openrouter
        self.model_name = model_name
        
        # Initialize LLM (OpenRouter or OpenAI)
        if OPENAI_AVAILABLE:
            try:
                if use_openrouter:
                    # Use OpenRouter API
                    self.llm = ChatOpenAI(
                        model=model_name,
                        temperature=0.7,
                        openai_api_key=self.api_key,
                        openai_api_base="https://openrouter.ai/api/v1",
                        default_headers={"HTTP-Referer": "https://github.com/your-repo"}  # Optional: for tracking
                    )
                else:
                    # Use direct OpenAI API
                    model = model_name.replace("openai/", "") if "openai/" in model_name else model_name
                    # Try both model and model_name for compatibility
                    try:
                        self.llm = ChatOpenAI(
                            model=model,
                            temperature=0.7,
                            openai_api_key=self.api_key
                        )
                    except:
                        self.llm = ChatOpenAI(
                            model_name=model,
                            temperature=0.7,
                            openai_api_key=self.api_key
                        )
            except Exception as e:
                raise ValueError(f"Failed to initialize LLM: {e}")
        else:
            raise ImportError("langchain-openai package is required. Install with: pip install langchain-openai")
        
        # Initialize drug interaction checker with RAG enabled
        self.checker = DrugInteractionChecker(data_path=data_path, use_rag=True, 
                                            openai_api_key=self.api_key, use_openrouter=use_openrouter,
                                            model_name=model_name)
        # Update RAG system to use API
        if self.checker.rag_system:
            self.checker.rag_system.api_key = self.api_key
            self.checker.rag_system.use_openai = True
            self.checker.rag_system.use_openrouter = use_openrouter
            self.checker.rag_system.model_name = model_name
            self.checker.rag_system._initialize_embeddings()
            self.checker.rag_system._initialize_llm()
        
        self.stats = self.checker.get_statistics()
        
        # System prompt
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the chatbot."""
        return f"""You are a helpful medical assistant chatbot specialized in drug-drug interactions. 
You have access to a database of {self.stats['num_interactions']} drug interactions covering {self.stats['num_drugs']} unique drugs.

Your capabilities:
1. Check for drug-drug interactions when given a list of medications
2. Explain interaction mechanisms and risks using RAG (Retrieval-Augmented Generation)
3. Provide safety recommendations based on interaction severity
4. Answer general questions about drug interactions

When a user provides medications, you should:
- Use the backend system to check for actual interactions
- Use RAG to retrieve relevant context from the database
- Provide clear, accurate information about any interactions found
- Include risk levels (High/Medium/Low) and severity information
- Offer practical safety recommendations
- Always remind users to consult healthcare professionals for medical advice

Be professional, clear, and helpful. Never provide definitive medical diagnoses or replace professional medical advice.
"""
    
    def check_drug_interactions(self, medications: List[str]) -> Dict:
        """
        Check for drug interactions using the backend with RAG.
        
        Args:
            medications: List of drug names
            
        Returns:
            Dictionary with interactions and formatted response
        """
        interactions = self.checker.check_interactions(medications)
        
        result = {
            'medications': medications,
            'interactions': interactions,
            'count': len(interactions)
        }
        
        return result
    
    def format_interactions_for_llm(self, interaction_result: Dict) -> str:
        """
        Format interaction results for LLM context.
        
        Args:
            interaction_result: Result from check_drug_interactions
            
        Returns:
            Formatted string for LLM
        """
        if interaction_result['count'] == 0:
            return f"No known interactions found among: {', '.join(interaction_result['medications'])}"
        
        formatted = f"Found {interaction_result['count']} interaction(s) among: {', '.join(interaction_result['medications'])}\n\n"
        
        for i, interaction in enumerate(interaction_result['interactions'], 1):
            formatted += f"Interaction {i}:\n"
            formatted += f"  Drugs: {interaction['drug1']} + {interaction['drug2']}\n"
            formatted += f"  Risk Level: {interaction.get('risk_level', 'Unknown')}\n"
            formatted += f"  Severity: {interaction.get('severity', 'Unknown')}\n"
            formatted += f"  Type: {interaction.get('interaction_type', 'Unknown')}\n"
            if interaction.get('mechanism'):
                formatted += f"  Mechanism: {interaction['mechanism']}\n"
            if interaction.get('description'):
                formatted += f"  Description: {interaction['description']}\n"
            if interaction.get('explanation'):
                formatted += f"  Explanation: {interaction['explanation']}\n"
            formatted += "\n"
        
        return formatted
    
    def chat(self, user_message: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Process a user message and generate a response using RAG.
        
        Args:
            user_message: User's message
            conversation_history: Previous conversation messages (optional)
            
        Returns:
            Bot's response
        """
        # Check if user is asking about specific medications
        medications = self._extract_medications(user_message)
        
        # If medications detected, check interactions first (this uses RAG)
        interaction_context = ""
        interaction_result = None
        if medications and len(medications) >= 2:
            try:
                interaction_result = self.check_drug_interactions(medications)
                interaction_context = self.format_interactions_for_llm(interaction_result)
            except Exception as e:
                interaction_context = f"Error checking interactions: {str(e)}"
        elif medications and len(medications) == 1:
            # Only one medication found
            interaction_context = f"Found medication: {medications[0]}. Please provide at least two medications to check for interactions."
        
        # Build prompt - be more direct to avoid placeholder responses
        if interaction_context and interaction_result:
            user_prompt = f"""Based on the drug interaction database check, I found the following:

{interaction_context}

Please provide a clear, direct answer about these drug interactions. Do NOT say "I will check" or "Just a moment" - the check is already done. Instead, directly explain:
1. What interactions were found (or if none were found)
2. The risk levels and severity
3. Mechanisms if available
4. Safety recommendations
5. Remind to consult healthcare professionals

Be direct and informative. Start your response immediately with the findings."""
        elif interaction_context:
            user_prompt = f"{interaction_context}\n\nUser question: {user_message}"
        else:
            user_prompt = f"User question: {user_message}\n\nPlease provide a helpful response about drug interactions."
        
        # Generate response using OpenAI/OpenRouter
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = response.content if hasattr(response, 'content') else str(response)
            
            # Check if response is a placeholder and replace with actual data
            placeholder_phrases = ["just a moment", "checking", "i will now check", "let me check", "please wait"]
            if any(phrase in result.lower() for phrase in placeholder_phrases):
                # Replace placeholder with actual interaction data
                if interaction_result and interaction_result.get('count', 0) > 0:
                    return self.format_interactions_for_llm(interaction_result) + "\n\n⚠️ **Important:** Always consult with a healthcare professional before making any medication decisions."
                elif interaction_result and interaction_result.get('count', 0) == 0:
                    return f"No known interactions found between {', '.join(medications)}.\n\n⚠️ **Note:** This does not guarantee safety. Always consult with a healthcare professional."
            
            return result
        except Exception as e:
            error_msg = str(e)
            # Provide fallback with interaction data if available
            if interaction_result and interaction_result.get('count', 0) > 0:
                return self.format_interactions_for_llm(interaction_result) + f"\n\n[Note: LLM generation encountered an issue, but here are the interaction results from the database]"
            elif medications and len(medications) >= 2:
                return f"Error generating LLM response: {error_msg}\n\nHowever, I can tell you that I checked the database for interactions between {', '.join(medications)}. Please try rephrasing your question or check the interactions directly."
            return f"Error generating response: {error_msg}\n\nNote: The backend interaction checker is still available. You can check interactions directly using the database."
    
    def _extract_medications(self, text: str) -> List[str]:
        """
        Enhanced medication extraction from text with common drug name aliases.
        Supports extraction of 2 or more medications.
        
        Args:
            text: User's message
            
        Returns:
            List of potential medication names (2 or more)
        """
        medications = []
        text_lower = text.lower()
        
        # Common drug name aliases mapping
        drug_aliases = {
            'aspirin': ['acetylsalicylic acid', 'asa', 'aspirin'],
            'warfarin': ['warfarin', 'coumadin'],
            'ibuprofen': ['ibuprofen', 'advil', 'motrin'],
            'acetaminophen': ['acetaminophen', 'paracetamol', 'tylenol'],
            'digoxin': ['digoxin', 'lanoxin'],
            'furosemide': ['furosemide', 'lasix'],
            'metformin': ['metformin', 'glucophage'],
            'lisinopril': ['lisinopril', 'prinivil', 'zestril'],
            'atorvastatin': ['atorvastatin', 'lipitor'],
            'amlodipine': ['amlodipine', 'norvasc'],
            'dolutegravir': ['dolutegravir'],
            'naltrexone': ['naltrexone'],
            'abacavir': ['abacavir'],
        }
        
        # Get all unique drugs from database for matching
        all_drugs = self.checker.data_loader.get_drugs()
        drug_lower_map = {drug.lower(): drug for drug in all_drugs}
        
        # First, try to match common aliases (handle multiple drugs)
        for alias_key, alias_list in drug_aliases.items():
            if alias_key in text_lower:
                # Try to find the actual drug name in database
                for alias in alias_list:
                    for db_drug in all_drugs:
                        if alias.lower() in db_drug.lower() or db_drug.lower() in alias.lower():
                            if db_drug not in medications:
                                medications.append(db_drug)
                                break
                    if medications and any(db_drug == medications[-1] for db_drug in all_drugs):
                        break
        
        # Also try direct word matching - improved for multiple drugs
        # Split by common conjunctions and separators
        separators = [',', 'and', '&', 'between', 'with', '+', 'plus']
        parts = text
        for sep in separators:
            parts = parts.replace(sep, '|')
        parts = parts.split('|')
        
        # Also split by spaces to catch individual words
        all_parts = []
        for part in parts:
            all_parts.extend(part.split())
        parts = all_parts
        
        # Check each word/phrase against known drugs
        for part in parts:
            part_clean = part.strip('.,!?;:()[]{}').lower()
            
            # Skip common words
            skip_words = ['check', 'interactions', 'between', 'drug', 'drugs', 'medication', 
                         'medications', 'interaction', 'and', 'with', 'for', 'the', 'a', 'an']
            if part_clean in skip_words or len(part_clean) < 3:
                continue
            
            # Exact match
            if part_clean in drug_lower_map:
                drug = drug_lower_map[part_clean]
                if drug not in medications:
                    medications.append(drug)
            else:
                # Partial match (contains) - more careful matching
                for db_drug_lower, db_drug in drug_lower_map.items():
                    # Check if the part is a significant portion of the drug name
                    if (part_clean in db_drug_lower and len(part_clean) >= 4) or \
                       (db_drug_lower in part_clean and len(db_drug_lower) >= 4):
                        if db_drug not in medications:
                            medications.append(db_drug)
                            break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_medications = []
        for med in medications:
            if med.lower() not in seen:
                seen.add(med.lower())
                unique_medications.append(med)
        
        return unique_medications

