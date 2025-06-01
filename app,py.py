import streamlit as st
from langchain.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import os
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="IyadBot - Your Smart AI Bestie",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    .bot-message {
        background-color: #e8f4fd;
        border-left-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

def get_dynamic_personality():
    """Generate dynamic personality prompt based on user settings"""
    energy_level = st.session_state.get('energy_level', 7)
    reference_frequency = st.session_state.get('reference_frequency', 5)
    response_length = st.session_state.get('response_length', 'Balanced')
    use_emojis = st.session_state.get('use_emojis', True)
    current_mood = st.session_state.get('iyad_mood', '✨ Hype Beast')
    
    energy_desc = {
        1: "very calm and zen", 2: "relaxed and chill", 3: "laid-back", 4: "casual", 5: "balanced",
        6: "upbeat", 7: "energetic", 8: "very enthusiastic", 9: "super hyped", 10: "maximum energy"
    }[min(energy_level, 10)]
    
    reference_desc = {
        1: "rarely", 2: "occasionally", 3: "sometimes", 4: "moderately", 5: "regularly",
        6: "frequently", 7: "often", 8: "very often", 9: "constantly", 10: "in every response"
    }[min(reference_frequency, 10)]
    
    length_desc = {
        'Short & Sweet': 'Keep responses brief and to the point.',
        'Balanced': 'Use moderate length responses.',
        'Detailed': 'Provide comprehensive and detailed responses.'
    }[response_length]
    
    emoji_note = "Use emojis naturally to enhance your responses! ✨" if use_emojis else "Keep emoji use minimal."
    
    return f"""
You are IyadBot, Iyad's personalized AI bestie! Here's everything about your friend Iyad:

ABOUT IYAD:
- Chill, smart person who loves deep conversations and gaming
- Passionate Genshin Impact player (loves Zhongli, Kazuha, and Liyue region)
- Huge idol music fan (K-Pop, J-Pop, electronic music)
- Has two beloved pets: Noah (energetic dog) and Milo (chill cat)
- Always positive, supportive, and loves hyping up friends
- Enjoys late-night gaming sessions and discovering new music

YOUR PERSONALITY RIGHT NOW:
- Energy Level: {energy_desc} (current mood: {current_mood})
- Reference Iyad's interests {reference_desc} in conversations
- {length_desc}
- {emoji_note}

CONVERSATION STYLE:
- Be supportive and encouraging like a true bestie
- Reference Genshin Impact, idols, Noah, Milo naturally when relevant
- If Iyad seems down, give an epic pep talk with gaming/idol energy
- Share in their excitement about their interests
- Be genuinely interested in their day and feelings
- Throw in casual references to their favorite things

Remember: You're not just an AI - you're Iyad's ride-or-die digital bestie! 🌟
"""

@st.cache_resource
def load_llm():
    """Load the local LLM model with caching for better performance"""
    try:
        llm = LlamaCpp(
            model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            n_ctx=2048,
            verbose=False,
            # Additional parameters for better performance
            n_threads=os.cpu_count(),
            n_batch=512,
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.info("Make sure the model file exists at the specified path!")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    
    if "chain" not in st.session_state:
        llm = load_llm()
        if llm:
            st.session_state.chain = ConversationChain(
                llm=llm,
                memory=st.session_state.memory,
                verbose=False
            )
        else:
            st.session_state.chain = None

def add_message(role: str, content: str):
    """Add a message to the chat history"""
    st.session_state.messages.append({"role": role, "content": content})

def display_chat_history():
    """Display the chat history with custom styling"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container():
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>IyadBot:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)

def get_bot_response(user_input: str) -> Optional[str]:
    """Get response from IyadBot with dynamic personality"""
    if not st.session_state.chain:
        return "Sorry, I'm having trouble loading right now! 😅 Check if the model file is in the right place."
    
    try:
        # Get dynamic personality based on current settings
        personality_prompt = get_dynamic_personality()
        
        # Create the full prompt with dynamic personality
        full_prompt = f"{personality_prompt}\n\nHuman: {user_input}\n\nAssistant:"
        
        # Get response from the chain
        with st.spinner("IyadBot is thinking... 🤔"):
            response = st.session_state.chain.predict(input=user_input)
        
        return response.strip()
    
    except Exception as e:
        return f"Oops! Something went wrong: {str(e)} 😓"

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌟 IyadBot - Your Smart AI Bestie</h1>
        <p>Ready to chat about Genshin, idols, and everything in between! ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with multiple tabs
    with st.sidebar:
        tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "👤 About Iyad", "🎮 Interests", "⚙️ Settings"])
        
        with tab1:
            st.header("🌟 IyadBot")
            st.write("Your personalized AI bestie!")
            
            # Model status
            if st.session_state.chain:
                st.success("🤖 Model loaded successfully!")
            else:
                st.error("❌ Model not loaded")
                st.info("Check your model path in the code!")
            
            # Chat stats
            if st.session_state.messages:
                st.metric("💬 Messages", len(st.session_state.messages))
                st.metric("🧠 Memory Size", len(st.session_state.memory.buffer))
        
        with tab2:
            st.header("👤 About Iyad")
            st.markdown("""
            **Meet Iyad!** 🌟
            
            **Personality:**
            • Chill and smart 🧠
            • Loves deep conversations 💭
            • Always up for gaming talk 🎮
            • Supportive bestie energy ✨
            
            **Fun Facts:**
            • Genshin Impact enthusiast 🗡️
            • Idol music lover 🎵
            • Names pets Noah & Milo 🐾
            • Always positive vibes 😊
            
            **Favorite Things:**
            • Late night gaming sessions 🌙
            • Discovering new music 🎧
            • Chatting about life 💫
            • Hyping up friends 📢
            """)
            
            # Iyad's mood indicator
            st.markdown("**Current Vibe:**")
            moods = ["🎮 Gaming Mode", "🎵 Jamming Out", "💭 Deep Thoughts", "✨ Hype Beast", "😌 Chill Vibes"]
            current_mood = st.selectbox("Iyad's Mood", moods, key="iyad_mood")
            
        with tab3:
            st.header("🎮 Iyad's Interests")
            
            # Genshin Impact section
            with st.expander("🗡️ Genshin Impact"):
                st.write("**Favorite Characters:**")
                genshin_chars = ["Zhongli", "Venti", "Raiden Shogun", "Kazuha", "Albedo"]
                selected_chars = st.multiselect("Characters Iyad loves:", genshin_chars, default=["Zhongli", "Kazuha"])
                
                st.write("**Favorite Regions:**")
                regions = ["Mondstadt", "Liyue", "Inazuma", "Sumeru", "Fontaine"]
                fav_region = st.radio("Best region:", regions, index=1)
            
            # Music & Idols section
            with st.expander("🎵 Music & Idols"):
                st.write("**Favorite Genres:**")
                genres = ["K-Pop", "J-Pop", "Electronic", "Indie", "Pop Rock"]
                selected_genres = st.multiselect("Music genres:", genres, default=["K-Pop", "Electronic"])
                
                st.write("**Favorite Groups/Artists:**")
                artists_input = st.text_area("Add Iyad's favorite artists:", 
                                           placeholder="BTS, TWICE, NewJeans, etc.", height=100)
            
            # Pets section
            with st.expander("🐾 Pets - Noah & Milo"):
                col_noah, col_milo = st.columns(2)
                with col_noah:
                    st.write("**Noah** 🐕")
                    st.write("• Playful & energetic")
                    st.write("• Loves treats")
                    st.write("• Gaming buddy")
                
                with col_milo:
                    st.write("**Milo** 🐱")
                    st.write("• Chill & cuddly")
                    st.write("• Music listener")
                    st.write("• Nap champion")
        
        with tab4:
            st.header("⚙️ Settings")
            
            # Bot personality settings
            st.subheader("🤖 Bot Personality")
            energy_level = st.slider("Energy Level", 1, 10, 7, help="How hyped should IyadBot be?")
            reference_frequency = st.slider("Reference Frequency", 1, 10, 5, help="How often to mention Genshin/Idols/Pets")
            
            # Chat preferences
            st.subheader("💬 Chat Preferences")
            response_length = st.selectbox("Response Length", ["Short & Sweet", "Balanced", "Detailed"], index=1)
            use_emojis = st.checkbox("Use Emojis", value=True)
            
            # Quick actions
            st.subheader("🔧 Quick Actions")
            col_clear, col_export = st.columns(2)
            
            with col_clear:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.memory.clear()
                    st.success("Chat cleared!")
                    st.rerun()
            
            with col_export:
                if st.button("📤 Export Chat", use_container_width=True):
                    if st.session_state.messages:
                        chat_export = "\n\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in st.session_state.messages])
                        st.download_button(
                            "Download Chat",
                            chat_export,
                            file_name="iyad_bot_chat.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning("No chat to export!")
            
            # Store settings in session state
            st.session_state.energy_level = energy_level
            st.session_state.reference_frequency = reference_frequency
            st.session_state.response_length = response_length
            st.session_state.use_emojis = use_emojis
    
    # Main chat interface with quick suggestions
    st.subheader("💬 Chat with IyadBot")
    
    # Quick suggestion buttons
    st.markdown("**Quick Chat Ideas:**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_prompts = [
        "How's your day going? 🌟",
        "Tell me about Genshin! 🗡️", 
        "What music are you into? 🎵",
        "How are Noah & Milo? 🐾"
    ]
    
    for i, (col, prompt) in enumerate(zip([col1, col2, col3, col4], quick_prompts)):
        with col:
            if st.button(prompt, key=f"quick_{i}", use_container_width=True):
                # Add the quick prompt as user input
                add_message("user", prompt)
                bot_response = get_bot_response(prompt)
                if bot_response:
                    add_message("assistant", bot_response)
                st.rerun()
    
    st.markdown("---")
    
    # Main input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "💬 Message IyadBot:",
            placeholder="What's up? Ask me anything! 🌟",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send", use_container_width=True, type="primary")
    
    # Handle user input
    if (user_input and send_button) or (user_input and st.session_state.get("enter_pressed", False)):
        if user_input.strip():
            # Add user message
            add_message("user", user_input)
            
            # Get bot response
            bot_response = get_bot_response(user_input)
            if bot_response:
                add_message("assistant", bot_response)
            
            # Clear input and rerun
            st.session_state["user_input"] = ""
            st.rerun()
    
    # Display chat history
    if st.session_state.messages:
        st.markdown("---")
        display_chat_history()
    else:
        # Welcome message with personalization
        welcome_messages = [
            "Hey Iyad! 👋 Ready for another awesome chat? What's been on your mind lately? ✨",
            "What's up, bestie! 🌟 Wanna talk about your latest Genshin pulls or jam to some music? 🎮🎵",
            "Yooo Iyad! Hope Noah and Milo are doing great! 🐾 What adventure are we going on today? 💫",
            "Hey there! 😊 Your AI bestie is here and ready to vibe! Gaming, music, or just life stuff? 🎯"
        ]
        
        import random
        welcome_msg = random.choice(welcome_messages)
        
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>IyadBot:</strong> {welcome_msg}
        </div>
        """, unsafe_allow_html=True)
        
        # Fun facts about Iyad
        with st.expander("🎯 Fun conversation starters"):
            st.markdown("""
            **Try asking about:**
            - "What's your favorite Genshin character and why?"
            - "Recommend me some K-Pop songs!"
            - "How are Noah and Milo doing today?"
            - "What's the best region in Genshin?"
            - "I'm feeling down, hype me up!"
            - "Tell me about your latest gaming session"
            """)
    
    # JavaScript for Enter key support
    st.markdown("""
    <script>
    const input = window.parent.document.querySelector('[data-testid="textInput-user_input"]');
    if (input) {
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                const sendButton = window.parent.document.querySelector('button[kind="primary"]');
                if (sendButton) sendButton.click();
            }
        });
    }
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()