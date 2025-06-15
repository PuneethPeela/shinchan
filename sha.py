import streamlit as st
import requests
import json
from typing import Dict, Any, List
import time

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #e8f4f8;
        margin-right: 20%;
    }
    .message-content {
        margin: 0;
        padding: 0;
    }
    .message-timestamp {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
    }
    .sidebar-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class HuggingFaceChatClient:
    """Client for interacting with Hugging Face Chat Assistant API"""
    
    def __init__(self, api_token: str, model_name: str = None):
        self.api_token = api_token
        self.model_name = model_name
        self.base_url = "https://api-inference.huggingface.co/models"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       max_tokens: int = 512, 
                       temperature: float = 0.7,
                       stream: bool = False) -> Dict[str, Any]:
        """Send chat completion request to Hugging Face API"""
        
        # If using a specific model endpoint
        if self.model_name:
            url = f"{self.base_url}/{self.model_name}"
        else:
            # Default to a popular chat model if none specified
            url = f"{self.base_url}/microsoft/DialoGPT-medium"
        
        # Format messages for the API
        if len(messages) > 0:
            # For most HF models, we need to format the conversation
            conversation = self._format_conversation(messages)
            
            payload = {
                "inputs": conversation,
                "parameters": {
                    "max_length": max_tokens,
                    "temperature": temperature,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
        else:
            payload = {"inputs": "Hello! How can I help you today?"}
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return {"response": result[0]["generated_text"]}
                elif "text" in result[0]:
                    return {"response": result[0]["text"]}
            elif isinstance(result, dict):
                if "generated_text" in result:
                    return {"response": result["generated_text"]}
                elif "text" in result:
                    return {"response": result["text"]}
            
            return {"response": "I'm here to help! What would you like to know?"}
            
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse API response: {str(e)}")
            return {"error": "Invalid API response"}
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation history for the API"""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                formatted += f"Human: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"
        
        # Add prompt for next response
        formatted += "Assistant:"
        return formatted

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_client" not in st.session_state:
        st.session_state.chat_client = None
    
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

def display_chat_message(message: Dict[str, str], timestamp: str = None):
    """Display a chat message with styling"""
    role = message["role"]
    content = message["content"]
    
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "🧑‍💻" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div class="message-content">
            <strong>{icon} {role.title()}:</strong><br>
            {content}
        </div>
        {f'<div class="message-timestamp">{timestamp}</div>' if timestamp else ''}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        st.title("🤖 Chat Configuration")
        
        # API Configuration
        st.subheader("API Settings")
        
        # Get API token from secrets or user input
        api_token = st.secrets.get("HF_API_TOKEN", "")
        if not api_token:
            api_token = st.text_input(
                "Hugging Face API Token",
                type="password",
                help="Enter your Hugging Face API token"
            )
        
        # Model selection
        model_options = {
            "Custom Model": "",
            "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "microsoft/DialoGPT-medium": "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large": "microsoft/DialoGPT-large",
            "facebook/blenderbot-400M-distill": "facebook/blenderbot-400M-distill",
            "microsoft/GODEL-v1_1-large-seq2seq": "microsoft/GODEL-v1_1-large-seq2seq",
            "bigscience/bloom-560m": "bigscience/bloom-560m"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0,
            help="Choose a pre-defined model or select 'Custom Model' to enter your own"
        )
        
        if selected_model == "Custom Model":
            model_name = st.text_input(
                "Custom Model Name",
                value=st.secrets.get("HF_MODEL_NAME", ""),
                help="Enter your custom model name (e.g., username/model-name)",
                key="custom_model_input"
            )
        else:
            model_name = model_options[selected_model]
            if model_name:  # Only show info if model_name is not empty
                st.info(f"Selected: {model_name}")
        
        # Chat parameters
        st.subheader("Chat Parameters")
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=1000, value=512)
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
        
        # Initialize chat client
        if api_token and st.button("Initialize Chat"):
            try:
                st.session_state.chat_client = HuggingFaceChatClient(
                    api_token=api_token,
                    model_name=model_name if model_name else None
                )
                st.session_state.model_loaded = True
                st.success("Chat assistant initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize chat assistant: {str(e)}")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Display model info
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
            if model_name:
                st.info(f"Using model: {model_name}")
            else:
                st.info("Using default model")
            
            # Show model description
            model_descriptions = {
                "Qwen/Qwen2.5-Coder-32B-Instruct": "🚀 Advanced coding assistant",
                "microsoft/DialoGPT-medium": "💬 Conversational AI",
                "microsoft/DialoGPT-large": "💬 Large conversational AI",
                "facebook/blenderbot-400M-distill": "🤖 Facebook's chatbot",
                "microsoft/GODEL-v1_1-large-seq2seq": "📝 Goal-oriented dialog",
                "bigscience/bloom-560m": "🌸 Multilingual language model"
            }
            
            if model_name in model_descriptions:
                st.caption(model_descriptions[model_name])
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main chat interface
    st.title("🤖 AI Chat Assistant")
    st.markdown("---")
    
    # Check if chat client is initialized
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please configure and initialize the chat assistant in the sidebar first.")
        st.info("""
        **Setup Instructions:**
        1. Enter your Hugging Face API token in the sidebar
        2. Optionally specify your custom model name
        3. Click 'Initialize Chat' to start
        
        **For Streamlit Cloud Deployment:**
        - Add your API token to Streamlit secrets as `HF_API_TOKEN`
        - Optionally add your model name as `HF_MODEL_NAME`
        """)
        return
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            display_chat_message(message)
    
    # Chat input
    st.markdown("---")
    
    # Create columns for input and send button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Your message:",
            placeholder="Type your message here...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Handle message sending
    if (send_button or user_input) and user_input.strip():
        # Add user message to chat history
        user_message = {"role": "user", "content": user_input.strip()}
        st.session_state.messages.append(user_message)
        
        # Show typing indicator
        with st.spinner("AI is thinking..."):
            try:
                # Get AI response
                response = st.session_state.chat_client.chat_completion(
                    messages=st.session_state.messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    # Add assistant response to chat history
                    assistant_message = {
                        "role": "assistant", 
                        "content": response.get("response", "I'm sorry, I couldn't generate a response.")
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Rerun to display the new message
                    st.rerun()
                    
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
