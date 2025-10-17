import argparse
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat
from agno.models.base import Model
import logging
from agno.models.ibm import WatsonX
import os
from dotenv import load_dotenv, find_dotenv, dotenv_values

MODEL_CONFIG = {

    "ollama:qwen2.5:latest": {"provider": "ollama", "api_key_required": False},
    "granite-code:8b": {"provider": "ollama", "api_key_required": False},
    "openai:gpt-4o": {"provider": "openai", "api_key_required": True},
    # Add more models here
}

logging.basicConfig(level=logging.INFO)



def validate_model_id(model_id: str) -> bool:
    """
    Validate the model_id against the expected format and available models.
    """
    if not isinstance(model_id, str) or ":" not in model_id:
        return False

    provider, model = model_id.split(":", 1)
    if provider.lower() not in [config["provider"] for config in MODEL_CONFIG.values()]:
        return False

    if model_id not in MODEL_CONFIG:
        return False

    return True






def get_model(model_id: str = None) -> Model:
    # Use find_dotenv() to automatically locate the nearest .env file
    dotenv_path = find_dotenv()
    env = {}
    
    if dotenv_path:
        logging.info(f"Loading environment variables from dotenv file: {dotenv_path}")
        load_dotenv(dotenv_path)
        env.update(dotenv_values(dotenv_path))
    
    # Also load from system environment as fallback
    env.update(os.environ)
    
    
    if model_id is None:
        return Ollama(id="qwen2.5:latest") # Default model

    if not validate_model_id(model_id):
        print(f"Invalid model ID: {model_id}. Falling back to default model.")
        return Ollama(id="qwen2.5:latest")
    
    provider, model = model_id.split(":", 1)


    match provider.lower():
        
        
        case "ollama":
            # Use OpenAIChat with a custom base_url to connect to a local Ollama server.
            try:
                # This avoids the need for the 'ollama' python package.
                # The API key can be any non-empty string for local Ollama.
                logging.info(f"Initializing Ollama model with model_id: {model_id}")
                return OpenAIChat(
                    id = model, api_key="ollama", base_url="http://127.0.0.1:11434/v1"
                )
            except Exception as e:
                logging.error(
                    f"Error initializing Ollama model {model_id}: {e}", exc_info=True
                )
                raise ValueError(f"Error initializing Ollama model: {e}") from e
        case "openai":
            if env.get("OPENAI_API_KEY") is None:
                logging.error("OPENAI_API_KEY environment variable is not set.")
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
            logging.info(f"Initializing OpenAI model with model_id: {model_id}")
            return OpenAIChat(id=model, api_key=env.get("OPENAI_API_KEY"))

        case "anthropic":
            if env.get("ANTHROPIC_API_KEY") is None:
                logging.error("ANTHROPIC_API_KEY environment variable is not set.")
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is required for Anthropic models"
                )
            try:
                from agno.models.anthropic import Anthropic
                return Anthropic(id=model, api_key=env.get("ANTHROPIC_API_KEY"))
            except ImportError as e:
                logging.error(
                    f"Error initializing Anthropic model {model_id}: {e}", exc_info=True
                )
                raise ImportError("Anthropic library not found. Please install it.") from e

        case "watsonx":
            if any(env.get(key) is None for key in ["IBM_WATSONX_API_KEY", "IBM_WATSONX_PROJECT_ID", "IBM_WATSONX_BASE_URL"]):
                raise ValueError("IBM_WATSONX_API_KEY, IBM_WATSONX_PROJECT_ID, and IBM_WATSONX_BASE_URL environment variables are required for WatsonX models")
            return WatsonX(
                id=model,
                url=env.get("IBM_WATSONX_BASE_URL"),
                api_key=env.get("IBM_WATSONX_API_KEY"),
                project_id=env.get("IBM_WATSONX_PROJECT_ID"),
            )
        case _:
            return Ollama(id="qwen2.5:latest")  # Default to Ollama



def create_cli_parser() -> argparse.ArgumentParser:
    """
    Create a command-line argument parser with common agent options.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Run an interactive agent CLI",)

    parser.add_argument(
        "--model-id", 
        type=str,
        default="openai:gpt-4o",
        help="Model identifier in the format 'provider:model'. Supported providers: "
             "ollama (e.g., ollama:qwen2.5:latest), "
             "openai (e.g., openai:gpt-4o), "
             "anthropic (e.g., anthropic:claude-3-sonnet), "
             "watsonx (e.g., watsonx:granite-13b)"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode for response generation"
    )
    
    return parser