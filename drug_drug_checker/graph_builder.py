"""
Graph builder for drug-drug interactions using NetworkX.
"""

import networkx as nx
from typing import List, Dict, Optional, Tuple
from data_loader import DDInterDataLoader


class DrugInteractionGraph:
    """Builds and manages a NetworkX graph of drug interactions."""
    
    def __init__(self, data_loader: Optional[DDInterDataLoader] = None):
        """
        Initialize graph builder.
        
        Args:
            data_loader: DDInterDataLoader instance. If None, creates a new one.
        """
        self.data_loader = data_loader or DDInterDataLoader()
        self.graph = nx.Graph()
        self._build_graph()
    
    def _build_graph(self):
        """Build the interaction graph from loaded data."""
        interactions = self.data_loader.get_interactions()
        
        for interaction in interactions:
            drug1 = interaction['drug1'].lower().strip()
            drug2 = interaction['drug2'].lower().strip()
            
            # Add edge with interaction data as attributes
            if drug1 != drug2:  # Avoid self-loops
                self.graph.add_edge(
                    drug1,
                    drug2,
                    interaction_type=interaction.get('interaction_type', 'Unknown'),
                    severity=interaction.get('severity', 'Moderate'),
                    mechanism=interaction.get('mechanism', ''),
                    description=interaction.get('description', '')
                )
    
    def has_interaction(self, drug1: str, drug2: str) -> bool:
        """Check if two drugs have an interaction."""
        d1 = drug1.lower().strip()
        d2 = drug2.lower().strip()
        return self.graph.has_edge(d1, d2)
    
    def get_interaction_data(self, drug1: str, drug2: str) -> Optional[Dict]:
        """Get interaction data for a drug pair."""
        d1 = drug1.lower().strip()
        d2 = drug2.lower().strip()
        
        if self.graph.has_edge(d1, d2):
            return self.graph[d1][d2]
        return None
    
    def find_interactions(self, drug_list: List[str]) -> List[Dict]:
        """
        Find all interactions among drugs in the given list.
        
        Args:
            drug_list: List of drug names to check
            
        Returns:
            List of interaction dictionaries with drug1, drug2, and interaction data
        """
        interactions = []
        normalized_drugs = [drug.lower().strip() for drug in drug_list]
        
        # Check all pairs
        for i, drug1 in enumerate(normalized_drugs):
            for drug2 in normalized_drugs[i+1:]:
                if self.has_interaction(drug1, drug2):
                    interaction_data = self.get_interaction_data(drug1, drug2)
                    # Find original casing for display
                    orig_drug1 = next((d for d in drug_list if d.lower().strip() == drug1), drug1)
                    orig_drug2 = next((d for d in drug_list if d.lower().strip() == drug2), drug2)
                    
                    interactions.append({
                        'drug1': orig_drug1,
                        'drug2': orig_drug2,
                        **interaction_data
                    })
        
        return interactions
    
    def get_risk_level(self, severity: str) -> str:
        """Map severity to risk level."""
        severity_lower = severity.lower()
        if 'contraindicated' in severity_lower or 'severe' in severity_lower:
            return 'High'
        elif 'major' in severity_lower:
            return 'High'
        elif 'moderate' in severity_lower:
            return 'Medium'
        elif 'minor' in severity_lower:
            return 'Low'
        else:
            return 'Medium'
    
    def get_graph(self) -> nx.Graph:
        """Get the NetworkX graph object."""
        return self.graph
    
    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        return {
            'num_drugs': self.graph.number_of_nodes(),
            'num_interactions': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph),
            'num_components': nx.number_connected_components(self.graph)
        }

