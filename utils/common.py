
import re
import json

from langchain_core.tools import tool

def extract_json(text: str) -> dict:
    """
    Estrae il primo oggetto JSON valido da un testo generato da LLM.
    Funziona anche se il testo è formattato con markdown o contiene altre parti non JSON.

    Args:
        text (str): Testo in output da un AIMessage

    Returns:
        dict: Dizionario Python risultante dal parsing del JSON

    Raises:
        ValueError: Se non riesce ad estrarre un JSON valido
    """
    # Regex per blocchi ```json ... ``` oppure {...}
    json_patterns = [
        r"```json\s*(\{.*?\})\s*```",  # blocco markdown ```json
        r"```[\s]*({.*?})[\s]*```",    # blocco markdown senza specificatore
        r"(\{.*\})"                    # prima struttura JSON valida
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    raise ValueError("❌ Nessun oggetto JSON valido trovato nel testo.")




