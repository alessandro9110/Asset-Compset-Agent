import os
import yaml
from pprint import pprint
import json
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import ValidationError


from utils.states import AgentState, PositionAnalysis
from langchain_core.messages import AIMessage
from utils.common import extract_json
from tools.initial_asset_assessment_tools import initial_asset_assessment_list


load_dotenv(override=True)
# Create model as brain
model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_MODEL"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))

with open("prompts/initial_asset_assessment_prompt.yaml", 'r') as stream:
    initial_asset_prompt = yaml.safe_load(stream)


# Create system message for Agents
sys_msg_initial_asset = SystemMessage(content=initial_asset_prompt['system_prompt'])



def initial_asset_assessment_agent(state: AgentState):

    # Costruisci la history dei messaggi
    messages = [sys_msg_initial_asset] + state["messages"]

    model_with_tools = model.bind_tools(initial_asset_assessment_list)
    # Invoca il modello
    result = model_with_tools.invoke(messages)
    messages.append(result)
    #print("🔍 Risultato del modello:")
    #print(result.content)


    
    # Torna lo stato aggiornato
    return {**state,
        "messages": messages,
        "position_analysis": result.content
    }

def initial_asset_assessment_output(state: dict) -> dict:
    """Estrae e formatta il risultato strutturato dal messaggio AI finale"""
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None

    if not isinstance(last_msg, AIMessage):
        raise ValueError("Ultimo messaggio non valido o mancante")

    try:
        result = extract_json(last_msg.content)
    except Exception as e:
        raise ValueError(f"Errore nel parsing JSON: {e}")

    return {
        **state,
        "initial_asset_assessment_result": result
    }