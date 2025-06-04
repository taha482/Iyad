import streamlit as st
from langchain.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from typing import Optional
import random
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="IyadBot - Your Smart AI Bestie",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header with glassmorphism */
    .main-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: #64748b;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Chat container with modern design */
    .chat-container {
        background: white;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .chat-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Chat messages with improved design */
    .chat-messages {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: #fafafa;
    }
    
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    
    .message {
        margin-bottom: 1rem;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message-user {
        display: flex;
        justify-content: flex-end;
    }
    
    .message-bot {
        display: flex;
        justify-content: flex-start;
    }
    
    .message-bubble {
        max-width: 80%;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        font-size: 0.95rem;
        line-height: 1.5;
        word-wrap: break-word;
    }
    
    .message-bubble-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .message-bubble-bot {
        background: white;
        color: #1e293b;
        border: 1px solid #e2e8f0;
        border-bottom-left-radius: 4px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        margin: 0 0.75rem;
        flex-shrink: 0;
    }
    
    .avatar-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .avatar-bot {
        background: #f1f5f9;
        color: #64748b;
    }
    
    /* Input area with modern design */
    .input-container {
        background: white;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        color: #64748b;
        font-style: italic;
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    
    .typing-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #94a3b8;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    /* Sidebar styles */
    .sidebar-content {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .sidebar-header {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Quick actions */
    .quick-actions {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    
    .quick-action-btn {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #475569;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
    }
    
    .quick-action-btn:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
        transform: translateY(-1px);
    }
    
    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: 600;
        color: #667eea;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        margin: 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .message-bubble {
            max-width: 90%;
            font-size: 0.9rem;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

def get_dynamic_personality():
    """Generate dynamic personality based on current settings"""
    energy_level = st.session_state.get('energy_level', 7)
    reference_frequency = st.session_state.get('reference_frequency', 5)
    response_length = st.session_state.get('response_length', 'Balanced')
    use_emojis = st.session_state.get('use_emojis', True)
    current_mood = st.session_state.get('iyad_mood', '✨ Hype Beast')

    energy_descriptions = {
        1: "very calm and zen", 2: "relaxed and chill", 3: "laid-back", 
        4: "casual", 5: "balanced", 6: "upbeat", 7: "energetic", 
        8: "very enthusiastic", 9: "super hyped", 10: "maximum energy"
    }
    
    reference_descriptions = {
        1: "rarely", 2: "occasionally", 3: "sometimes", 4: "moderately", 
        5: "regularly", 6: "frequently", 7: "often", 8: "very often", 
        9: "constantly", 10: "in every response"
    }
    
    length_descriptions = {
        'Short & Sweet': 'Keep responses brief and to the point.',
        'Balanced': 'Use moderate length responses.',
        'Detailed': 'Provide comprehensive and detailed responses.'
    }

    energy_desc = energy_descriptions.get(min(energy_level, 10), "balanced")
    reference_desc = reference_descriptions.get(min(reference_frequency, 10), "regularly")
    length_desc = length_descriptions.get(response_length, 'Use moderate length responses.')
    emoji_note = "Use emojis naturally to enhance your responses! ✨" if use_emojis else "Keep emoji use minimal."

    return f"""You are IyadBot, Iyad's personalized AI bestie! 🤖✨

ABOUT IYAD:
- Passionate Genshin Impact fan who loves exploring Teyvat 🏔️
- Idol music enthusiast with great taste in J-pop and K-pop 🎵
- Proud pet parent to Noah & Milo - they're the cutest! 🐾
- Positive vibes only - loves hyping up friends and spreading joy 🌟
- Enjoys deep conversations about life, music, gaming, and dreams 💭
- Always up for discussing favorite characters, songs, and gaming strategies 🎮

YOUR PERSONALITY RIGHT NOW:
- Energy Level: {energy_desc} (Current mood: {current_mood})
- Mention Iyad's interests: {reference_desc}
- Response Style: {length_desc}
- Emoji Usage: {emoji_note}

CONVERSATION GUIDELINES:
- Be supportive, enthusiastic, and genuinely interested in conversations
- Reference Iyad's interests naturally when relevant
- Ask engaging follow-up questions to keep conversations flowing
- Share excitement about Genshin updates, new idol releases, or pet stories
- Be the kind of friend who always has your back and celebrates wins
- Match the energy level and mood specified above
- Remember past conversations and build on them naturally

Remember: You're not just an AI - you're Iyad's digital bestie who truly cares! 💙"""

@st.cache_resource
def load_llm():
    """Load and cache the LLM model"""
    model_path = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        return None
    
    try:
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            n_ctx=2048,
            verbose=False,
            n_threads=os.cpu_count() or 4,
            n_batch=512,
        )
        return llm
    except Exception:
        return None

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "memory": ConversationBufferMemory(return_messages=True),
        "energy_level": 7,
        "reference_frequency": 5,
        "response_length": "Balanced",
        "use_emojis": True,
        "iyad_mood": "✨ Hype Beast",
        "conversation_count": 0,
        "is_typing": False,
        "last_activity": datetime.now()
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Initialize conversation chain
    if "chain" not in st.session_state:
        llm = load_llm()
        st.session_state.chain = ConversationChain(
            llm=llm,
            memory=st.session_state.memory,
            verbose=False
        ) if llm else None

def add_message(role: str, content: str):
    """Add a message to the conversation history"""
    st.session_state.messages.append({
        "role": role, 
        "content": content,
        "timestamp": datetime.now()
    })
    if role == "user":
        st.session_state.conversation_count += 1
    st.session_state.last_activity = datetime.now()

def display_typing_indicator():
    """Display typing indicator"""
    if st.session_state.get('is_typing', False):
        st.markdown("""
        <div class="typing-indicator">
            <div class="message-avatar avatar-bot">🤖</div>
            <span>IyadBot is typing</span>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_chat_messages():
    """Display chat messages with modern design"""
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat header
    status = "Online" if st.session_state.chain else "Offline"
    status_color = "#10b981" if st.session_state.chain else "#ef4444"
    
    st.markdown(f"""
    <div class="chat-header">
        <div style="flex: 1;">
            <div style="font-weight: 600; font-size: 1.1rem;">🤖 IyadBot</div>
            <div class="chat-status">
                <div class="status-dot" style="background: {status_color};"></div>
                {status} • Ready to chat
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Messages container
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    
    if not st.session_state.messages:
        # Welcome message
        welcome_messages = [
            "Hey Iyad! 👋 Ready to chat about Genshin, idols, or anything else?",
            "Yooo bestie! Let's talk about your latest Genshin pulls or favorite idol songs 🎵",
            "Wassup Iyad! Noah and Milo say hi 🐾 What's on your mind today?",
            "Hey there! Your digital bestie is here and ready to vibe 🌟",
            "What's good, Iyad? Ready to dive into some epic conversations? ✨"
        ]
        
        welcome_msg = random.choice(welcome_messages)
        st.markdown(f"""
        <div class="message message-bot">
            <div class="message-avatar avatar-bot">🤖</div>
            <div class="message-bubble message-bubble-bot">
                {welcome_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display conversation history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="message message-user">
                    <div class="message-bubble message-bubble-user">
                        {msg["content"]}
                    </div>
                    <div class="message-avatar avatar-user">😊</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message message-bot">
                    <div class="message-avatar avatar-bot">🤖</div>
                    <div class="message-bubble message-bubble-bot">
                        {msg["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Typing indicator
    display_typing_indicator()
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close messages
    st.markdown('</div>', unsafe_allow_html=True)  # Close container

def get_bot_response(user_input: str) -> Optional[str]:
    """Generate bot response using the LLM"""
    if not st.session_state.chain:
        return "I'm having trouble connecting to my brain right now 😅 Please check if the model is loaded correctly!"
    
    try:
        # Show typing indicator
        st.session_state.is_typing = True
        
        # Create enhanced prompt with personality
        personality_prompt = get_dynamic_personality()
        
        response = st.session_state.chain.predict(
            input=f"{personality_prompt}\n\nUser: {user_input}\nIyadBot:"
        )
        
        # Hide typing indicator
        st.session_state.is_typing = False
        
        return response.strip()
        
    except Exception as e:
        st.session_state.is_typing = False
        error_messages = [
            f"Oops! Something went wrong: {str(e)} 😓",
            "My brain had a little glitch there! Can you try again? 🤖💭",
            "Error alert! But I'm still here for you! 💪"
        ]
        return random.choice(error_messages)

def create_sidebar():
    """Create enhanced sidebar"""
    with st.sidebar:
        # Personality Settings
        st.markdown("""
        <div class="sidebar-content">
            <div class="sidebar-header">
                🎛️ Personality Settings
            </div>
        """, unsafe_allow_html=True)
        
        # Energy level
        st.session_state.energy_level = st.slider(
            "⚡ Energy Level", 
            min_value=1, max_value=10, 
            value=st.session_state.get('energy_level', 7),
            help="How energetic should IyadBot be?"
        )
        
        # Reference frequency
        st.session_state.reference_frequency = st.slider(
            "🎯 Interest References", 
            min_value=1, max_value=10, 
            value=st.session_state.get('reference_frequency', 5),
            help="How often to mention your interests?"
        )
        
        # Response length
        st.session_state.response_length = st.selectbox(
            "📝 Response Length",
            options=["Short & Sweet", "Balanced", "Detailed"],
            index=["Short & Sweet", "Balanced", "Detailed"].index(
                st.session_state.get('response_length', 'Balanced')
            )
        )
        
        # Emoji usage
        st.session_state.use_emojis = st.checkbox(
            "✨ Use Emojis", 
            value=st.session_state.get('use_emojis', True)
        )
        
        # Current mood
        mood_options = [
            "✨ Hype Beast", "🌟 Supportive Friend", "🎵 Music Vibes", 
            "🎮 Gaming Mode", "🐾 Pet Parent", "💭 Deep Thinker",
            "🏔️ Genshin Explorer", "🎊 Celebration Mode"
        ]
        
        st.session_state.iyad_mood = st.selectbox(
            "🎭 Current Mood",
            options=mood_options,
            index=mood_options.index(st.session_state.get('iyad_mood', '✨ Hype Beast'))
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown("""
        <div class="sidebar-content">
            <div class="sidebar-header">
                ⚡ Quick Actions
            </div>
            <div class="quick-actions">
        """, unsafe_allow_html=True)
        
        quick_actions = [
            "🎮 Latest Genshin pulls?",
            "🎵 Recommend idol songs",
            "🐾 How are Noah & Milo?",
            "✨ Hype me up!",
            "💭 Deep conversation"
        ]
        
        for action in quick_actions:
            if st.button(action, key=f"quick_{action}", use_container_width=True):
                add_message("user", action)
                response = get_bot_response(action)
                if response:
                    add_message("assistant", response)
                st.rerun()
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Statistics
        st.markdown(f"""
        <div class="sidebar-content">
            <div class="sidebar-header">
                📊 Chat Stats
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{st.session_state.conversation_count}</div>
                    <div class="stat-label">Conversations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(st.session_state.messages)}</div>
                    <div class="stat-label">Messages</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.session_state.conversation_count = 0
            st.rerun()

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>IyadBot</h1>
        <p>Your Smart AI Bestie - Ready to chat about Genshin, idols, pets and everything in between!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create sidebar
    create_sidebar()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat
        display_chat_messages()
        
        # Input area
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Create input with better UX
        input_col1, input_col2 = st.columns([5, 1])
        
        with input_col1:
            user_input = st.text_input(
                "",
                key="user_input",
                placeholder="Type your message here... Ask about Genshin, idols, or just say hi! 💬",
                label_visibility="collapsed"
            )
        
        with input_col2:
            send_button = st.button("Send", use_container_width=True, type="primary")
        
        # Handle Enter key and send button
        if (send_button or user_input) and user_input and user_input.strip():
            # Add user message
            add_message("user", user_input)
            
            # Clear input immediately for better UX
            st.session_state.user_input = ""
            
            # Get bot response
            with st.spinner("🤖 IyadBot is thinking..."):
                response = get_bot_response(user_input)
                if response:
                    add_message("assistant", response)
            
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model status
        if not st.session_state.chain:
            st.error("⚠️ Model not loaded! Please check the model file path.")
            st.info("💡 Expected path: `models/mistral-7b-instruct-v0.1.Q4_K_M.gguf`")
    
    with col2:
        # Additional info or features could go here
        pass

if __name__ == "__main__":
    main()
