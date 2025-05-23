from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config.settings as settings
import config.prompts as prompts

def get_chain(model_name=None, temperature=None):
    """
    Creates a RAG chain with the specified model and temperature, using defaults from settings if not provided.
    """
    # Use provided values or fall back to defaults from settings
    effective_model_name = model_name if model_name is not None else settings.DEFAULT_MODEL_NAME
    effective_temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
    
    if not settings.GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in settings.")
        return None
        
    try:
        model = ChatGroq(
            temperature=effective_temperature,
            groq_api_key=settings.GROQ_API_KEY,
            model_name=effective_model_name,
            max_tokens=None,  # Let model decide appropriate response length
            timeout=None,     # No timeout limit
            max_retries=2,    # Retry failed requests twice
        )
    except Exception as e:
        print(f"Error initializing ChatGroq: {e}")
        return None

    parser = StrOutputParser()
    prompt_template = ChatPromptTemplate.from_template(prompts.RAG_SYSTEM_PROMPT)
    chain = prompt_template | model | parser
    return chain

def get_fast_chain(temperature=None):
    """
    Creates a RAG chain with a smaller, faster model for quick initial responses.
    """
    effective_temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
    
    if not settings.GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in settings.")
        return None
        
    try:
        # Use the fastest model for initial responses
        fast_model = ChatGroq(
            temperature=effective_temperature,
            groq_api_key=settings.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",  # Latest fast model
            max_tokens=150,   # Limit tokens for faster preview
            timeout=None,
            max_retries=1,    # Single retry for fast response
        )
    except Exception as e:
        print(f"Error initializing fast ChatGroq: {e}")
        return None

    parser = StrOutputParser()
    # Use a more concise prompt for the fast model
    prompt_template = ChatPromptTemplate.from_template(prompts.RAG_FAST_PROMPT)
    chain = prompt_template | fast_model | parser
    return chain

def ask_question(chain, question, context, conversation_history: list = None):
    """
    Generate a response using the provided chain, context, and history.
    Handles potential chain errors.
    
    Args:
        chain: The Langchain chain object.
        question: The user's question.
        context: The retrieved context relevant to the question.
        conversation_history: A list of dictionaries representing the history 
                              (e.g., [{'role': 'Human', 'content': '...'}, {'role': 'Assistant', 'content': '...'}]). 
                              Defaults to None for the start of a conversation.
                              
    Returns:
        A tuple containing:
            - response (str): The generated answer or an error message.
            - updated_history (list): The updated conversation history list.
    """
    if not chain:
        print("Error: ask_question called with an invalid chain object.")
        # Return an empty history list on chain error if none was provided
        return "Error: Chain not initialized.", conversation_history or []
        
    # Initialize history if it's the first turn
    history = conversation_history or []
        
    try:
        # The current RAG prompt expects {context} and {question}.
        # History is managed here but not passed directly to this specific prompt template.
        formatted_input = {"context": context, "question": question}
        
        response = chain.invoke(formatted_input)
        
        updated_history = history + [
            {"role": "Human", "content": question},
            {"role": "Assistant", "content": response},
        ]
        
        return response, updated_history
        
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        error_message = f"Sorry, I encountered an error trying to answer: {e}"
        # Append the error interaction to history
        updated_history = history + [
            {"role": "Human", "content": question},
            {"role": "Assistant", "content": error_message},
        ]
        return error_message, updated_history