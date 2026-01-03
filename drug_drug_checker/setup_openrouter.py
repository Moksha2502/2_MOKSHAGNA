"""
Quick setup script to configure OpenRouter API key.
"""

import os

print("="*60)
print("Drug Interaction Chatbot - OpenRouter API Key Setup")
print("="*60)

api_key = input("\nEnter your OpenRouter API key (or press Enter to skip): ").strip()

if api_key:
    # Create .env file
    env_content = f"OPENROUTER_API_KEY={api_key}\n"
    
    if os.path.exists('.env'):
        print("\n.env file already exists. Updating OPENROUTER_API_KEY...")
        # Read existing .env
        with open('.env', 'r') as f:
            lines = f.readlines()
        
        # Update or add OPENROUTER_API_KEY
        updated = False
        with open('.env', 'w') as f:
            for line in lines:
                if line.startswith('OPENROUTER_API_KEY='):
                    f.write(env_content)
                    updated = True
                else:
                    f.write(line)
            if not updated:
                f.write(env_content)
    else:
        print("\nCreating .env file...")
        with open('.env', 'w') as f:
            f.write(env_content)
    
    print("\n[OK] API key configured successfully!")
    print("You can now run: streamlit run streamlit_app.py")
else:
    print("\nSkipped. You can set the API key in the Streamlit app sidebar or set OPENROUTER_API_KEY environment variable.")

print("="*60)

