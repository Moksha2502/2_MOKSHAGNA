"""
Main Drug-Drug Interaction Checker class.
Combines graph-based detection with RAG-powered explanations.
"""

from typing import List, Dict, Optional
from data_loader import DDInterDataLoader
from graph_builder import DrugInteractionGraph
from rag_system import RAGSystem


class DrugInteractionChecker:
    """Main class for checking drug-drug interactions."""
    
    def __init__(self, data_path: Optional[str] = None, use_rag: bool = True, 
                 openai_api_key: Optional[str] = None, use_openrouter: bool = False,
                 model_name: str = "openai/gpt-3.5-turbo"):
        """
        Initialize the drug interaction checker.
        
        Args:
            data_path: Path to DDInter CSV file. If None, uses sample data.
            use_rag: Whether to use RAG system for enhanced explanations.
            openai_api_key: OpenRouter or OpenAI API key for RAG system. If None, tries to get from environment.
            use_openrouter: Whether to use OpenRouter API (True) or direct OpenAI (False).
            model_name: Model to use for LLM (e.g., "openai/gpt-3.5-turbo" for OpenRouter).
        """
        self.data_loader = DDInterDataLoader(data_path)
        self.graph = DrugInteractionGraph(self.data_loader)
        self.rag_system = None
        
        if use_rag:
            interactions_data = self.data_loader.get_interactions()
            self.rag_system = RAGSystem(interactions_data, api_key=openai_api_key, 
                                      use_openai=True, use_openrouter=use_openrouter, 
                                      model_name=model_name)
    
    def check_interactions(self, medications: List[str]) -> List[Dict]:
        """
        Check for interactions among a list of medications.
        
        Args:
            medications: List of drug names to check
            
        Returns:
            List of interaction dictionaries with:
            - drug1, drug2: Drug names
            - interaction_type: Type of interaction
            - severity: Severity level
            - risk_level: Risk level (High/Medium/Low)
            - mechanism: Interaction mechanism
            - description: Interaction description
            - explanation: RAG-generated explanation
        """
        # Find interactions using graph
        interactions = self.graph.find_interactions(medications)
        
        # Enhance with RAG explanations
        enhanced_interactions = []
        for interaction in interactions:
            # Add risk level
            interaction['risk_level'] = self.graph.get_risk_level(
                interaction.get('severity', 'Moderate')
            )
            
            # Enhance with RAG if available
            if self.rag_system:
                interaction = self.rag_system.enhance_interaction(interaction)
            else:
                # Fallback explanation
                interaction['explanation'] = interaction.get('description', '') or interaction.get('mechanism', 'No explanation available.')
            
            enhanced_interactions.append(interaction)
        
        # Sort by risk level (High -> Medium -> Low)
        risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
        enhanced_interactions.sort(
            key=lambda x: risk_order.get(x.get('risk_level', 'Medium'), 1)
        )
        
        return enhanced_interactions
    
    def get_statistics(self) -> Dict:
        """Get statistics about the interaction graph."""
        return self.graph.get_statistics()
    
    def visualize_graph(self, save_path: Optional[str] = None):
        """
        Visualize the interaction graph.
        
        Args:
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            graph = self.graph.get_graph()
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                                 node_size=500, alpha=0.9)
            
            # Draw edges with color based on severity
            edge_colors = []
            for (u, v, d) in graph.edges(data=True):
                severity = d.get('severity', 'Moderate').lower()
                if 'major' in severity or 'contraindicated' in severity:
                    edge_colors.append('red')
                elif 'moderate' in severity:
                    edge_colors.append('orange')
                else:
                    edge_colors.append('gray')
            
            nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, 
                                 alpha=0.6, width=2)
            
            # Draw labels
            nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')
            
            plt.title('Drug-Drug Interaction Graph', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Graph visualization saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available for visualization")
        except Exception as e:
            print(f"Error creating visualization: {e}")

