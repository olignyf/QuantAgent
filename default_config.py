DEFAULT_CONFIG = {
    "agent_llm_model": "qwen3.5:9b-120k",
    "graph_llm_model": "qwen3.5:9b-120k",
    "agent_llm_provider": "ollama",  # "openai", "anthropic", "qwen", or "ollama"
    "graph_llm_provider": "ollama",
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "api_key": "sk-",  # OpenAI API key
    "anthropic_api_key": "sk-",  # Anthropic API key (optional, can also use ANTHROPIC_API_KEY env var)
    "qwen_api_key": "sk-",  # Qwen API key (optional, can also use DASHSCOPE_API_KEY env var)
    # Ollama: OpenAI-compatible API (default local). API key is often ignored; use "ollama" if required.
    "ollama_base_url": "http://localhost:11434/v1",
    "ollama_api_key": "ollama",
}
