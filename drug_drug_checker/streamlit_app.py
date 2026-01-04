"""
Streamlit frontend for Drug-Drug Interaction Chatbot with visualizations.
"""

import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from dotenv import load_dotenv
from chatbot_openai import DrugInteractionChatbot, OPENAI_AVAILABLE

# Load environment variables
try:
    load_dotenv(encoding='utf-8')
except UnicodeDecodeError:
    try:
        import codecs
        if os.path.exists('.env'):
            with codecs.open('.env', 'r', encoding='utf-16') as f:
                content = f.read()
            with open('.env', 'w', encoding='utf-8') as f:
                f.write(content)
            load_dotenv(encoding='utf-8')
        else:
            load_dotenv()
    except:
        load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Drug Interaction Analyzer",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        padding-bottom: 1rem;
        font-size: 1.1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #fee;
        border-left-color: #d32f2f;
    }
    .risk-medium {
        background-color: #fff8e1;
        border-left-color: #f57c00;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left-color: #388e3c;
    }
    .drug-info-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_interactions' not in st.session_state:
    st.session_state.current_interactions = []

def initialize_chatbot():
    """Initialize the chatbot with API key from environment."""
    try:
        api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
        # API key is optional if we're in template mode
        # if not api_key:
        #     return None, "API key not found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable."
        
        data_path = 'data/ddinter_downloads_code_A.csv'
        if not os.path.exists(data_path):
            data_path = None
        
        # Initialize chatbot - it will work in template mode even if LLM fails
        chatbot = DrugInteractionChatbot(
            api_key=api_key, 
            data_path=data_path, 
            use_openrouter=True, 
            model_name="openai/gpt-3.5-turbo"
        )
        return chatbot, None
    except Exception as e:
        return None, str(e)

# Auto-initialize chatbot
if not st.session_state.chatbot:
    chatbot, error = initialize_chatbot()
    if chatbot:
        st.session_state.chatbot = chatbot
    elif error:
        st.error(f"⚠️ {error}")
        st.stop()

# Show warnings if features are unavailable, but continue anyway
if not OPENAI_AVAILABLE:
    st.warning("⚠️ **LLM Features Unavailable**: langchain-openai is not available (possibly due to PyTorch DLL error). "
               "The application will continue in template-based mode. Drug interaction checking will still work, "
               "but AI-generated explanations are disabled. To enable AI features, install Visual C++ Redistributables "
               "and reinstall PyTorch.")

# Check if chatbot has LLM available
if st.session_state.chatbot and hasattr(st.session_state.chatbot, 'llm_available') and not st.session_state.chatbot.llm_available:
    st.info("ℹ️ **Template Mode Active**: The application is running in template-based mode. "
            "Drug interaction checking works perfectly, but AI-generated explanations are disabled. "
            "If you have an API key, check that it's valid and set correctly. See TROUBLESHOOTING.md for help.")

# Main header
st.markdown('<div class="main-header">💊 Drug-Drug Interaction Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Drug Interaction Detection with Visual Analytics</div>', unsafe_allow_html=True)

# Statistics banner
if st.session_state.chatbot:
    stats = st.session_state.chatbot.stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Drugs", f"{stats.get('num_drugs', 0):,}")
    with col2:
        st.metric("🔗 Interactions", f"{stats.get('num_interactions', 0):,}")
    with col3:
        st.metric("📈 Graph Density", f"{stats.get('density', 0):.4f}")
    with col4:
        st.metric("🔷 Components", stats.get('num_components', 0))

st.divider()

# Side effects database (common side effects for common interactions)
SIDE_EFFECTS_DB = {
    'bleeding': ['Increased bleeding risk', 'Gastrointestinal bleeding', 'Bruising', 'Nosebleeds'],
    'cardiovascular': ['Hypotension', 'Bradycardia', 'Arrhythmias', 'QT prolongation'],
    'renal': ['Acute kidney injury', 'Increased creatinine', 'Decreased urine output'],
    'hepatic': ['Liver toxicity', 'Elevated liver enzymes', 'Hepatitis'],
    'metabolic': ['Hyperkalemia', 'Hypokalemia', 'Hypoglycemia', 'Hyperglycemia'],
    'neurological': ['Dizziness', 'Confusion', 'Sedation', 'Seizures'],
    'gastrointestinal': ['Nausea', 'Vomiting', 'Diarrhea', 'GI ulcers'],
}

def get_side_effects_for_interaction(interaction):
    """Get relevant side effects based on interaction type and severity."""
    severity = interaction.get('severity', '').lower()
    mechanism = interaction.get('mechanism', '').lower()
    description = interaction.get('description', '').lower()
    
    side_effects = []
    
    # Check for bleeding-related interactions
    if any(term in mechanism + description for term in ['bleeding', 'anticoagulant', 'platelet', 'coagulation']):
        side_effects.extend(SIDE_EFFECTS_DB['bleeding'])
    
    # Check for cardiovascular
    if any(term in mechanism + description for term in ['cardiac', 'heart', 'qt', 'hypotension', 'bradycardia']):
        side_effects.extend(SIDE_EFFECTS_DB['cardiovascular'])
    
    # Check for renal
    if any(term in mechanism + description for term in ['kidney', 'renal', 'creatinine', 'nephro']):
        side_effects.extend(SIDE_EFFECTS_DB['renal'])
    
    # Check for hepatic
    if any(term in mechanism + description for term in ['liver', 'hepatic', 'hepatitis']):
        side_effects.extend(SIDE_EFFECTS_DB['hepatic'])
    
    # Check for metabolic
    if any(term in mechanism + description for term in ['potassium', 'kalemia', 'glucose', 'sugar']):
        side_effects.extend(SIDE_EFFECTS_DB['metabolic'])
    
    # Default side effects based on severity
    if 'major' in severity or 'contraindicated' in severity:
        if not side_effects:
            side_effects = ['Severe adverse effects', 'Toxicity', 'Serious complications']
    elif 'moderate' in severity:
        if not side_effects:
            side_effects = ['Moderate adverse effects', 'Dose-dependent toxicity']
    
    return list(set(side_effects))[:5]  # Return unique side effects, max 5

def create_severity_chart(interactions):
    """Create a pie chart of interaction severities."""
    if not interactions:
        return None
    
    severity_counts = Counter([i.get('severity', 'Unknown') for i in interactions])
    fig = px.pie(
        values=list(severity_counts.values()),
        names=list(severity_counts.keys()),
        title="Interaction Severity Distribution",
        color_discrete_map={
            'Major': '#d32f2f',
            'Moderate': '#f57c00',
            'Minor': '#388e3c',
            'Unknown': '#757575'
        }
    )
    fig.update_layout(showlegend=True, height=300)
    return fig

def create_risk_chart(interactions):
    """Create a bar chart of risk levels."""
    if not interactions:
        return None
    
    risk_counts = Counter([i.get('risk_level', 'Unknown') for i in interactions])
    fig = px.bar(
        x=list(risk_counts.keys()),
        y=list(risk_counts.values()),
        title="Risk Level Distribution",
        color=list(risk_counts.keys()),
        color_discrete_map={
            'High': '#d32f2f',
            'Medium': '#f57c00',
            'Low': '#388e3c'
        }
    )
    fig.update_layout(showlegend=False, height=300, xaxis_title="Risk Level", yaxis_title="Count")
    return fig

# Chat interface
col_left, col_right = st.columns([2, 1])

with col_left:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Enhanced interaction display
            if message["role"] == "assistant" and "interactions" in message:
                for interaction in message["interactions"]:
                    risk_level = interaction.get('risk_level', 'Unknown')
                    risk_class = f"risk-{risk_level.lower()}" if risk_level != 'Unknown' else ""
                    
                    st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"### {interaction['drug1']} + {interaction['drug2']}")
                    with col_b:
                        st.markdown(f"**Risk:** {risk_level}")
                    
                    # Drug information
                    st.markdown('<div class="drug-info-card">', unsafe_allow_html=True)
                    
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.write(f"**Severity:** {interaction.get('severity', 'Unknown')}")
                        st.write(f"**Type:** {interaction.get('interaction_type', 'Unknown')}")
                    
                    with info_col2:
                        st.write(f"**Risk Level:** {risk_level}")
                    
                    # Mechanism and description
                    if interaction.get('mechanism'):
                        st.markdown(f"**Mechanism:** {interaction['mechanism']}")
                    
                    if interaction.get('description'):
                        st.markdown(f"**Description:** {interaction['description']}")
                    
                    # Side effects
                    side_effects = get_side_effects_for_interaction(interaction)
                    if side_effects:
                        st.markdown("**⚠️ Potential Side Effects:**")
                        st.markdown(", ".join([f"• {se}" for se in side_effects]))
                    
                    # Explanation
                    if interaction.get('explanation'):
                        with st.expander("📋 Detailed Explanation"):
                            st.markdown(interaction['explanation'])
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")

with col_right:
    # Visualizations
    if st.session_state.current_interactions:
        st.subheader("📊 Visualizations")
        
        # Severity chart
        severity_fig = create_severity_chart(st.session_state.current_interactions)
        if severity_fig:
            st.plotly_chart(severity_fig, use_container_width=True)
        
        # Risk chart
        risk_fig = create_risk_chart(st.session_state.current_interactions)
        if risk_fig:
            st.plotly_chart(risk_fig, use_container_width=True)
        
        # Interaction summary table
        if st.session_state.current_interactions:
            st.subheader("📋 Interaction Summary")
            summary_data = []
            for i, interaction in enumerate(st.session_state.current_interactions, 1):
                summary_data.append({
                    'Drugs': f"{interaction['drug1']} + {interaction['drug2']}",
                    'Risk': interaction.get('risk_level', 'Unknown'),
                    'Severity': interaction.get('severity', 'Unknown'),
                    'Type': interaction.get('interaction_type', 'Unknown')
                })
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

# Chat input
if prompt := st.chat_input("Ask about drug interactions (e.g., 'Check interactions between Warfarin, Aspirin, and Ibuprofen')..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing interactions..."):
            # Extract medications
            medications = st.session_state.chatbot._extract_medications(prompt)
            
            # Get interaction data
            interactions_data = []
            if medications and len(medications) >= 2:
                try:
                    interaction_result = st.session_state.chatbot.check_drug_interactions(medications)
                    interactions_data = interaction_result['interactions']
                    st.session_state.current_interactions = interactions_data
                except Exception as e:
                    st.error(f"Error checking interactions: {str(e)}")
            
            # Generate chatbot response
            try:
                conversation_history = [
                    {"role": msg["role"], "parts": [msg["content"]]} 
                    for msg in st.session_state.messages[:-1]
                ]
                
                response = st.session_state.chatbot.chat(prompt, conversation_history)
                st.markdown(response)
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                if interactions_data:
                    st.info("Here are the interaction results from the database:")
            
            # Store in session state
            message_data = {"role": "assistant", "content": response}
            if interactions_data:
                message_data["interactions"] = interactions_data
            st.session_state.messages.append(message_data)
            
            # Rerun to update visualizations
            st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>⚠️ This tool is for informational purposes only. Always consult healthcare professionals for medical advice.</small>
</div>
""", unsafe_allow_html=True)

# Minimal sidebar (collapsed by default)
with st.sidebar:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.current_interactions = []
        st.rerun()
