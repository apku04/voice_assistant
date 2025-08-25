# voice_assistant/ollama_client.py
import requests
import logging
from typing import Dict, Any, List
from .config import config

# Set up logging
logger = logging.getLogger(__name__)

def call_ollama(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call Ollama API with fallback to different endpoints"""
    for url in (config.pc_ollama_url, config.pi_ollama_url):
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"Ollama endpoint {url} failed: {e}")
            continue
            
    raise RuntimeError("No Ollama server reachable (PC or Pi).")
    
# Add better error handling to the ollama_chat function
def _ollama_chat(system: str, user: str, model: str | None = None, temperature: float = 0.2) -> str:
    """Enhanced with better error handling for memory operations"""
    try:
        payload = {
            "model": model or config.default_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 2048},
        }
        data = call_ollama(payload)
        return data.get("message", {}).get("content", "") or ""
    except Exception as e:
        logger.warning(f"Ollama chat failed for memory operation: {e}")
        return ""  # Return empty string instead of crashing

def chat_once(history: List[Dict[str, str]], sys_prompt: str, user_text: str,
              current_model: str, *, deterministic: bool = False) -> Dict[str, Any]:
    """Send a single chat request to Ollama - matches original timing behavior"""
    # Create a copy to avoid mutating the original list
    local_history = history.copy()
    local_history.append({"role": "user", "content": user_text})
    
    options = {"num_ctx": 2048}
    options.update(
        {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 42} 
        if deterministic else 
        {"temperature": 0.4}
    )
    
    payload = {
        "model": current_model, 
        "messages": [{"role": "system", "content": sys_prompt}] + local_history,
        "stream": False, 
        "options": options
    }
    
    # Match original timing: busy indicator starts inside this function
    data = call_ollama(payload)
    reply = data["message"]["content"].strip()
    local_history.append({"role": "assistant", "content": reply})
    
    return {"reply": reply, "history": local_history}