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
        
        # Enhanced system prompts for different model types
        self.system_prompts = {
            "microsoft/DialoGPT-medium": "You are a helpful, friendly, and engaging conversational AI assistant. Provide thoughtful, coherent responses while maintaining a natural conversation flow.",
            "microsoft/DialoGPT-large": "You are a sophisticated conversational AI assistant. Engage in meaningful dialogue, provide detailed and helpful responses, and maintain context throughout the conversation.",
            "facebook/blenderbot-400M-distill": "You are BlenderBot, a knowledgeable and empathetic conversational AI. Provide informative, engaging responses while being personable and understanding.",
            "microsoft/GODEL-v1_1-large-seq2seq": "You are GODEL, a goal-oriented dialog assistant. Help users achieve their objectives through clear, structured, and actionable responses.",
            "bigscience/bloom-560m": "You are BLOOM, a multilingual AI assistant. Provide helpful, accurate, and culturally aware responses in the user's preferred language.",
            "Qwen/Qwen2.5-0.5B-Instruct": "You are Qwen, a helpful AI assistant created by Alibaba Cloud. Provide accurate, detailed, and contextually appropriate responses to help users with their questions and tasks.",
            "meta-llama/Llama-2-7b-chat-hf": "You are Llama, a helpful and harmless AI assistant. Provide thoughtful, accurate responses while being respectful and following ethical guidelines.",
            "mistralai/Mistral-7B-Instruct-v0.1": "You are Mistral, an AI assistant designed to be helpful, harmless, and honest. Provide clear, accurate, and useful responses to user queries.",
            "google/flan-t5-large": "You are Flan-T5, a text-to-text AI model. Provide helpful and accurate responses by understanding the user's intent and generating appropriate text.",
            "EleutherAI/gpt-j-6b": "You are GPT-J, a large language model. Provide helpful, creative, and contextually appropriate responses while maintaining a conversational tone.",
            "stabilityai/stablelm-tuned-alpha-7b": "You are StableLM, a helpful AI assistant. Provide clear, informative, and engaging responses while being reliable and trustworthy.",
            "default": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user queries while maintaining a friendly and professional tone."
        }
    
    def _get_model_specific_params(self, model_name: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Get model-specific parameters for optimal performance"""
        base_params = {
            "temperature": temperature,
            "do_sample": True,
            "return_full_text": False
        }
        
        # Model-specific parameter optimization
        if "DialoGPT" in model_name:
            return {
                **base_params,
                "max_length": min(max_tokens, 1000),
                "pad_token_id": 50256,
                "eos_token_id": 50256,
                "repetition_penalty": 1.1,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 2
            }
        elif "blenderbot" in model_name.lower():
            return {
                **base_params,
                "max_length": min(max_tokens, 512),
                "num_beams": 4,
                "early_stopping": True,
                "repetition_penalty": 1.2
            }
        elif "GODEL" in model_name:
            return {
                **base_params,
                "max_length": min(max_tokens, 512),
                "num_beams": 2,
                "repetition_penalty": 1.1,
                "length_penalty": 0.8
            }
        elif "bloom" in model_name.lower():
            return {
                **base_params,
                "max_new_tokens": min(max_tokens, 256),
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1
            }
        elif "Qwen" in model_name:
            return {
                **base_params,
                "max_new_tokens": min(max_tokens, 512),
                "top_p": 0.8,
                "top_k": 50,
                "repetition_penalty": 1.05
            }
        elif "Llama" in model_name or "llama" in model_name.lower():
            return {
                **base_params,
                "max_new_tokens": min(max_tokens, 512),
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.1
            }
        elif "Mistral" in model_name or "mistral" in model_name.lower():
            return {
                **base_params,
                "max_new_tokens": min(max_tokens, 512),
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.05
            }
        elif "flan-t5" in model_name.lower():
            return {
                **base_params,
                "max_length": min(max_tokens, 512),
                "num_beams": 4,
                "early_stopping": True,
                "repetition_penalty": 1.1
            }
        elif "gpt-j" in model_name.lower():
            return {
                **base_params,
                "max_new_tokens": min(max_tokens, 512),
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1
            }
        elif "stablelm" in model_name.lower():
            return {
                **base_params,
                "max_new_tokens": min(max_tokens, 512),
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.05
            }
        else:
            # Default parameters for unknown models
            return {
                **base_params,
                "max_length": max_tokens,
                "top_p": 0.9,
                "top_k": 50
            }
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       max_tokens: int = 512, 
                       temperature: float = 0.7,
                       stream: bool = False) -> Dict[str, Any]:
        """Send chat completion request to Hugging Face API with enhanced error handling"""
        
        # Validate model name
        if not self.model_name or len(self.model_name.strip()) < 3:
            return {"error": "Invalid model name provided"}
        
        # Clean model name
        model_name = self.model_name.strip()
        url = f"{self.base_url}/{model_name}"
        
        # Format messages for the API with enhanced system prompt
        if len(messages) > 0:
            conversation = self._format_conversation_with_system_prompt(messages, model_name)
            
            # Get model-specific parameters
            parameters = self._get_model_specific_params(model_name, max_tokens, temperature)
            
            payload = {
                "inputs": conversation,
                "parameters": parameters,
                "options": {
                    "wait_for_model": True,
                    "use_cache": False
                }
            }
        else:
            payload = {"inputs": "Hello! How can I help you today?"}
        
        try:
            st.info(f"Making request to: {url}")  # Debug info
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            
            # Enhanced error handling
            if response.status_code == 400:
                error_detail = response.json().get('error', 'Bad request')
                return {"error": f"Bad request: {error_detail}. Please check your input parameters."}
            elif response.status_code == 401:
                return {"error": "Authentication failed. Please check your API token."}
            elif response.status_code == 403:
                return {"error": "Access denied. Please verify your API token permissions or model access rights."}
            elif response.status_code == 404:
                return {"error": f"Model '{model_name}' not found. Please verify the model name exists on Hugging Face Hub."}
            elif response.status_code == 429:
                return {"error": "Rate limit exceeded. Please wait a moment before making another request."}
            elif response.status_code == 500:
                return {"error": "Internal server error. The model service is temporarily unavailable."}
            elif response.status_code == 503:
                retry_after = response.headers.get('Retry-After', '60')
                return {"error": f"Model is currently loading or unavailable. Please try again in {retry_after} seconds."}
            elif response.status_code == 504:
                return {"error": "Request timeout. The model took too long to respond. Try reducing max_tokens or simplifying your input."}
            
            # Check for successful response
            if not response.ok:
                return {"error": f"HTTP {response.status_code}: {response.reason}"}
            
            result = response.json()
            
            # Enhanced response processing
            return self._process_model_response(result, model_name)
            
        except requests.exceptions.Timeout:
            return {"error": "Request timed out. The model service is taking too long to respond."}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection error. Please check your internet connection and try again."}
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse API response. The service may be temporarily unavailable."}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def _format_conversation_with_system_prompt(self, messages: List[Dict[str, str]], model_name: str) -> str:
        """Format conversation history with enhanced system prompt for the API"""
        
        # Get appropriate system prompt
        system_prompt = self.system_prompts.get(model_name, self.system_prompts["default"])
        
        formatted = f"System: {system_prompt}\n\n"
        
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
    
    def _process_model_response(self, result: Any, model_name: str) -> Dict[str, Any]:
        """Enhanced response processing for different model types"""
        
        try:
            # Handle list responses (most common)
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                
                if isinstance(first_result, dict):
                    # Check for generated_text
                    if "generated_text" in first_result:
                        response_text = first_result["generated_text"]
                        return {"response": self._clean_response(response_text, model_name)}
                    
                    # Check for text field
                    elif "text" in first_result:
                        response_text = first_result["text"]
                        return {"response": self._clean_response(response_text, model_name)}
                    
                    # Check for error in response
                    elif "error" in first_result:
                        return {"error": first_result["error"]}
                
                # Handle string response in list
                elif isinstance(first_result, str):
                    return {"response": self._clean_response(first_result, model_name)}
            
            # Handle dictionary responses
            elif isinstance(result, dict):
                if "generated_text" in result:
                    response_text = result["generated_text"]
                    return {"response": self._clean_response(response_text, model_name)}
                elif "text" in result:
                    response_text = result["text"]
                    return {"response": self._clean_response(response_text, model_name)}
                elif "error" in result:
                    return {"error": result["error"]}
            
            # Handle direct string response
            elif isinstance(result, str):
                return {"response": self._clean_response(result, model_name)}
            
            # Fallback response
            return {"response": "I'm here to help! What would you like to know?"}
            
        except Exception as e:
            return {"error": f"Error processing model response: {str(e)}"}
    
    def _clean_response(self, response_text: str, model_name: str) -> str:
        """Clean and format the response text based on model type"""
        
        if not response_text:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        # Remove common artifacts and clean up response
        cleaned = response_text.strip()
        
        # Remove system prompt remnants
        if cleaned.startswith("System:"):
            parts = cleaned.split("Assistant:", 1)
            if len(parts) > 1:
                cleaned = parts[1].strip()
        
        # Remove conversation markers that might leak through
        cleaned = cleaned.replace("Human:", "").replace("Assistant:", "").strip()
        
        # Model-specific cleaning
        if "DialoGPT" in model_name:
            # Remove special tokens that might appear
            cleaned = cleaned.replace("<|endoftext|>", "").strip()
        
        elif "blenderbot" in model_name.lower():
            # BlenderBot sometimes repeats the input
            lines = cleaned.split('\n')
            if len(lines) > 1:
                cleaned = lines[-1].strip()
        
        elif "Qwen" in model_name:
            # Remove Qwen-specific tokens
            cleaned = cleaned.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        
        elif "Llama" in model_name or "llama" in model_name.lower():
            # Remove Llama-specific tokens
            cleaned = cleaned.replace("[INST]", "").replace("[/INST]", "").strip()
        
        # Remove excessive whitespace and empty lines
        cleaned = '\n'.join(line.strip() for line in cleaned.split('\n') if line.strip())
        
        # Ensure we have a meaningful response
        if not cleaned or len(cleaned) < 3:
            return "I understand your message, but I'm having trouble formulating a response right now. Could you please try asking in a different way?"
        
        return cleaned

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
            "microsoft/DialoGPT-medium": "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large": "microsoft/DialoGPT-large", 
            "facebook/blenderbot-400M-distill": "facebook/blenderbot-400M-distill",
            "microsoft/GODEL-v1_1-large-seq2seq": "microsoft/GODEL-v1_1-large-seq2seq",
            "bigscience/bloom-560m": "bigscience/bloom-560m",
            "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
            "meta-llama/Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
            "google/flan-t5-large": "google/flan-t5-large",
            "EleutherAI/gpt-j-6b": "EleutherAI/gpt-j-6b",
            "stabilityai/stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",
            "Custom Model": "custom"
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
            if not model_name.strip():
                st.warning("Please enter a valid model name")
        else:
            model_name = model_options[selected_model]
            st.info(f"Selected: {model_name}")
            
        # Validate model name
        if model_name and "/" not in model_name and selected_model != "Custom Model":
            st.error("Invalid model format. Model name should be in format: username/model-name")
            model_name = ""
        
        # Chat parameters
        st.subheader("Chat Parameters")
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=1000, value=512)
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
        
        # Initialize chat client
        if api_token and model_name and st.button("Initialize Chat"):
            try:
                # Validate model name format
                if selected_model != "Custom Model" or (selected_model == "Custom Model" and "/" in model_name):
                    st.session_state.chat_client = HuggingFaceChatClient(
                        api_token=api_token,
                        model_name=model_name
                    )
                    st.session_state.model_loaded = True
                    st.success("Chat assistant initialized successfully!")
                else:
                    st.error("Please provide a valid model name in format: username/model-name")
            except Exception as e:
                st.error(f"Failed to initialize chat assistant: {str(e)}")
                st.session_state.model_loaded = False
        
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
                "microsoft/DialoGPT-medium": "💬 Conversational AI",
                "microsoft/DialoGPT-large": "💬 Large conversational AI",
                "facebook/blenderbot-400M-distill": "🤖 Facebook's chatbot",
                "microsoft/GODEL-v1_1-large-seq2seq": "📝 Goal-oriented dialog",
                "bigscience/bloom-560m": "🌸 Multilingual language model",
                "Qwen/Qwen2.5-0.5B-Instruct": "🚀 Alibaba's Qwen model",
                "meta-llama/Llama-2-7b-chat-hf": "🦙 Meta's Llama 2 chat model",
                "mistralai/Mistral-7B-Instruct-v0.1": "🌟 Mistral instruction model",
                "google/flan-t5-large": "🔧 Google's Flan-T5 model",
                "EleutherAI/gpt-j-6b": "⚡ EleutherAI's GPT-J model",
                "stabilityai/stablelm-tuned-alpha-7b": "🎯 Stability AI's StableLM"
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
