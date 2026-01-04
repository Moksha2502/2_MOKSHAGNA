"""
Command-line interface for Drug-Drug Interaction Checker.
"""

import argparse
import sys
from typing import List
from drug_interaction_checker import DrugInteractionChecker


def format_interaction_output(interaction: dict) -> str:
    """Format interaction dictionary for display."""
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"[!] INTERACTION DETECTED: {interaction['drug1']} + {interaction['drug2']}")
    output.append(f"{'='*60}")
    output.append(f"Risk Level: {interaction.get('risk_level', 'Unknown')}")
    output.append(f"Severity: {interaction.get('severity', 'Unknown')}")
    output.append(f"Type: {interaction.get('interaction_type', 'Unknown')}")
    output.append(f"\nMechanism:")
    output.append(f"  {interaction.get('mechanism', 'Not specified')}")
    output.append(f"\nExplanation:")
    
    explanation = interaction.get('explanation', 'No explanation available.')
    # Wrap explanation text
    words = explanation.split()
    line = []
    for word in words:
        if len(' '.join(line + [word])) > 55:
            output.append(f"  {' '.join(line)}")
            line = [word]
        else:
            line.append(word)
    if line:
        output.append(f"  {' '.join(line)}")
    
    output.append("")
    return "\n".join(output)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Drug-Drug Interaction Checker (Graph + RAG)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --meds "Warfarin,Aspirin,Ibuprofen"
  python main.py --meds "Warfarin" "Aspirin" "Ibuprofen" --data data/ddinter.csv
  python main.py --meds "Digoxin,Furosemide" --stats
  python main.py --meds "Sildenafil,Nitrates" --visualize
        """
    )
    
    parser.add_argument(
        '--meds',
        nargs='+',
        required=True,
        help='List of medications to check (space-separated or comma-separated)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/ddinter_downloads_code_A.csv',
        help='Path to DDInter CSV data file (default: data/ddinter_downloads_code_A.csv, uses sample data if not found)'
    )
    
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable RAG system (use basic explanations)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show graph statistics'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate and display interaction graph visualization'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save visualization to file (requires --visualize)'
    )
    
    args = parser.parse_args()
    
    # Parse medications (handle both comma-separated and space-separated)
    medications = []
    for med_arg in args.meds:
        if ',' in med_arg:
            medications.extend([m.strip() for m in med_arg.split(',')])
        else:
            medications.append(med_arg.strip())
    
    medications = [m for m in medications if m]  # Remove empty strings
    
    if len(medications) < 2:
        print("Error: Please provide at least 2 medications to check.")
        sys.exit(1)
    
    print(f"\n[*] Checking interactions among {len(medications)} medications:")
    print(f"   {', '.join(medications)}\n")
    
    # Initialize checker
    try:
        checker = DrugInteractionChecker(data_path=args.data, use_rag=not args.no_rag)
    except Exception as e:
        print(f"Error initializing checker: {e}")
        sys.exit(1)
    
    # Show statistics if requested
    if args.stats:
        stats = checker.get_statistics()
        print("\n[STATS] Graph Statistics:")
        print(f"   Number of drugs in database: {stats['num_drugs']}")
        print(f"   Number of interactions: {stats['num_interactions']}")
        print(f"   Graph density: {stats['density']:.4f}")
        print(f"   Connected components: {stats['num_components']}")
        print()
    
    # Check for interactions
    try:
        interactions = checker.check_interactions(medications)
    except Exception as e:
        print(f"Error checking interactions: {e}")
        sys.exit(1)
    
    # Display results
    if interactions:
        print(f"\n[!] Found {len(interactions)} interaction(s):")
        for interaction in interactions:
            print(format_interaction_output(interaction))
    else:
        print("\n[OK] No known interactions detected among the provided medications.")
        print("   (Note: This does not guarantee safety - always consult healthcare providers)")
    
    # Visualize if requested
    if args.visualize:
        try:
            checker.visualize_graph(save_path=args.output)
        except Exception as e:
            print(f"\nWarning: Could not generate visualization: {e}")


if __name__ == '__main__':
    main()

