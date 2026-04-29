from langchain_ollama import ChatOllama


def get_llm(model: str = "mistral", temperature: float = 0.7):
    """
    Returns a ChatOllama instance ready for tool calling via bind_tools().
    ChatOllama (vs OllamaLLM) is required for structured tool/function calling.
    """
    return ChatOllama(model=model, temperature=temperature)


def get_llm_with_tools(tools: list, model: str = "mistral", temperature: float = 0.7):
    """Returns an LLM with the given tools bound for function calling."""
    llm = get_llm(model=model, temperature=temperature)
    return llm.bind_tools(tools)
