import os
import yaml
from pprint import pprint
import logging
import json
import re
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import ValidationError


from utils.states import AgentState
from langchain_core.messages import AIMessage
from utils.common import extract_json_fallback
from tools.initial_asset_analysis_tools import initial_asset_analysis_list

logging.basicConfig(
    filename='agent_log.txt',  # log su file (oppure usa filename=None per solo console)
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


load_dotenv(override=True)

# Create model as brain
model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_MODEL"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0)

# LOAD PROMPTS: base prompt + sections
with open("prompts/system_prompt_base.yaml", 'r') as stream:
    prompt_base = yaml.safe_load(stream)

with open("prompts/fase_1/position_analysis.yaml", 'r') as stream:
    prompt_position_analysis = yaml.safe_load(stream)


def run_section(state: AgentState, section:str, prompt_base: str, prompt_section:str):
    """ The Agent is resposible to create an initial Asset Analysis, following:
        - Geographic Location Assessment

        
    """

    prompt_base    = SystemMessage(content=prompt_base)
    prompt_section = SystemMessage(content=prompt_section)

    # Costruisci la history dei messaggi
    local_messages = [prompt_base,
                      prompt_section,
                      *state["messages"]]

    # Assegna i tool
    model_with_tools = model.bind_tools(initial_asset_analysis_list)

    # Invoca il modello
    result = model_with_tools.invoke(local_messages)

    # new history
    new_history = [*state["messages"], result]
    result = extract_json_fallback(result.content)


    out = {"messages": new_history}
    out[section] = result
    
    # Torna lo stato aggiornato
    return out


def initial_asset_analysis_agent(state: AgentState):
    """ The Agent is resposible to create an initial Asset Analysis, following:
        - Geographic Location Assessment

        
    """
    
    # Section 1: Position Analysis
    sec1 = run_section(state = state,
                       section = "position_analysis",
                       prompt_base = prompt_base['system_prompt'], 
                       prompt_section = prompt_position_analysis['system_prompt'])
    
    state = {**state, **sec1}
    
    
    # Torna lo stato aggiornato
    return state




def initial_asset_analysys_output(state: dict) -> dict:
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