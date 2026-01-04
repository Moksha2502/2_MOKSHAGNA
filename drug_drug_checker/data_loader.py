"""
Data loader for DDInter (Drug-Drug Interactions) dataset.
Handles loading and parsing of drug interaction data.
"""

import pandas as pd
import os
from typing import List, Dict, Optional


class DDInterDataLoader:
    """Loads and processes DDInter dataset."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to DDInter CSV file. If None, uses sample data.
        """
        self.data_path = data_path
        self.interactions: List[Dict] = []
    
    def load_from_csv(self, file_path: str) -> List[Dict]:
        """
        Load interactions from CSV file.
        Supports multiple formats:
        - DDInter format: Drug_A, Drug_B, Level
        - Extended format: Drug1, Drug2, Interaction_Type, Severity, Mechanism, Description
        """
        try:
            df = pd.read_csv(file_path)
            
            # Handle DDInter format (Drug_A, Drug_B, Level)
            if 'Drug_A' in df.columns and 'Drug_B' in df.columns:
                drug1_col = 'Drug_A'
                drug2_col = 'Drug_B'
                level_col = 'Level' if 'Level' in df.columns else None
            else:
                # Handle different possible column names for other formats
                drug1_col = next((col for col in df.columns if 'drug1' in col.lower() or 'drug_1' in col.lower() or col.lower() == 'drug a'), df.columns[0])
                drug2_col = next((col for col in df.columns if 'drug2' in col.lower() or 'drug_2' in col.lower() or col.lower() == 'drug b'), df.columns[1])
                level_col = next((col for col in df.columns if 'level' in col.lower() or 'severity' in col.lower() or 'risk' in col.lower()), None)
            
            interactions = []
            for _, row in df.iterrows():
                # Get drug names
                drug1 = str(row[drug1_col]).strip()
                drug2 = str(row[drug2_col]).strip()
                
                # Skip if drugs are empty or the same
                if not drug1 or not drug2 or drug1 == drug2:
                    continue
                
                # Get severity/level
                if level_col and level_col in row:
                    severity = str(row[level_col]).strip()
                else:
                    severity = str(row.get('Severity', row.get('Risk', 'Moderate'))).strip()
                
                # For DDInter format, we don't have mechanism/description in the CSV
                # So we'll use generic descriptions based on severity
                interaction = {
                    'drug1': drug1,
                    'drug2': drug2,
                    'interaction_type': str(row.get('Interaction_Type', row.get('Type', 'Unknown'))).strip(),
                    'severity': severity,
                    'mechanism': str(row.get('Mechanism', row.get('Mechanism_Description', 'Interaction mechanism not specified in dataset'))).strip(),
                    'description': str(row.get('Description', row.get('Details', f'Drug interaction between {drug1} and {drug2} with {severity} severity level'))).strip(),
                }
                interactions.append(interaction)
            
            self.interactions = interactions
            print(f"Loaded {len(interactions)} interactions from {file_path}")
            return interactions
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def create_sample_data(self) -> List[Dict]:
        """
        Create sample drug interaction data for testing.
        Based on common known interactions.
        """
        sample_interactions = [
            {
                'drug1': 'Warfarin',
                'drug2': 'Aspirin',
                'interaction_type': 'Pharmacodynamic',
                'severity': 'Major',
                'mechanism': 'Increased bleeding risk due to additive anticoagulant effects',
                'description': 'Both drugs inhibit platelet function and coagulation, significantly increasing the risk of bleeding.'
            },
            {
                'drug1': 'Warfarin',
                'drug2': 'Ibuprofen',
                'interaction_type': 'Pharmacodynamic',
                'severity': 'Major',
                'mechanism': 'Increased bleeding risk; NSAIDs can cause GI bleeding',
                'description': 'Ibuprofen increases bleeding risk and can cause gastrointestinal ulcers, which is dangerous with warfarin.'
            },
            {
                'drug1': 'Digoxin',
                'drug2': 'Furosemide',
                'interaction_type': 'Pharmacokinetic',
                'severity': 'Moderate',
                'mechanism': 'Hypokalemia from furosemide increases digoxin toxicity risk',
                'description': 'Furosemide causes potassium loss, which makes the heart more sensitive to digoxin, increasing toxicity risk.'
            },
            {
                'drug1': 'ACE Inhibitor',
                'drug2': 'Potassium Supplements',
                'interaction_type': 'Pharmacodynamic',
                'severity': 'Moderate',
                'mechanism': 'Increased risk of hyperkalemia',
                'description': 'ACE inhibitors reduce potassium excretion, and combined with potassium supplements, can cause dangerous hyperkalemia.'
            },
            {
                'drug1': 'Metformin',
                'drug2': 'Contrast Dye',
                'interaction_type': 'Pharmacokinetic',
                'severity': 'Major',
                'mechanism': 'Increased risk of lactic acidosis and contrast-induced nephropathy',
                'description': 'Contrast dye can cause kidney injury, which increases metformin accumulation and risk of lactic acidosis.'
            },
            {
                'drug1': 'Lithium',
                'drug2': 'Thiazide Diuretics',
                'interaction_type': 'Pharmacokinetic',
                'severity': 'Major',
                'mechanism': 'Reduced lithium clearance leading to toxicity',
                'description': 'Thiazide diuretics reduce lithium excretion by the kidneys, leading to increased blood levels and toxicity risk.'
            },
            {
                'drug1': 'Sildenafil',
                'drug2': 'Nitrates',
                'interaction_type': 'Pharmacodynamic',
                'severity': 'Contraindicated',
                'mechanism': 'Severe hypotension due to synergistic vasodilation',
                'description': 'Both drugs cause vasodilation. Combined use can cause severe, life-threatening hypotension.'
            },
            {
                'drug1': 'Theophylline',
                'drug2': 'Ciprofloxacin',
                'interaction_type': 'Pharmacokinetic',
                'severity': 'Major',
                'mechanism': 'Ciprofloxacin inhibits theophylline metabolism',
                'description': 'Ciprofloxacin inhibits CYP1A2 enzyme that metabolizes theophylline, leading to increased theophylline levels and toxicity.'
            },
        ]
        
        self.interactions = sample_interactions
        return sample_interactions
    
    def load(self) -> List[Dict]:
        """
        Load interactions from file or create sample data.
        """
        if self.data_path and os.path.exists(self.data_path):
            return self.load_from_csv(self.data_path)
        else:
            print(f"Data file not found at {self.data_path}. Using sample data.")
            return self.create_sample_data()
    
    def get_interactions(self) -> List[Dict]:
        """Get loaded interactions."""
        if not self.interactions:
            self.load()
        return self.interactions
    
    def get_drugs(self) -> List[str]:
        """Get unique list of all drugs in the dataset."""
        if not self.interactions:
            self.load()
        
        drugs = set()
        for interaction in self.interactions:
            drugs.add(interaction['drug1'])
            drugs.add(interaction['drug2'])
        return sorted(list(drugs))

