
import re
import json

from langchain_core.tools import tool

def extract_json_fallback(text):
    """
    Tries to extract a JSON object from the given text.
    Returns a dict if successful, else an empty dict.
    """
    try:
        # Primo tentativo: parsing diretto
        return json.loads(text)
    except Exception:
        pass

    # Fallback: cerca il primo oggetto tra { ... }
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except Exception:
            pass
    return {}



