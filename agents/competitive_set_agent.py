import os
import yaml
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI

load_dotenv(override=True)

from utils.states import AgentState
from tools.initial_asset_assessment_tools import *

# Create model as brain
model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_MODEL"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))

with open("prompts/initial_asset_assessment_prompt.yaml", 'r') as stream:
    competitive_set_prompt = yaml.safe_load(stream)


# Create system message for Agents
sys_msg_competitive_set = SystemMessage(content=competitive_set_prompt['system_prompt'])



def competitive_set_agent(state: AgentState):
   messages = [model.invoke([sys_msg_competitive_set] + state["messages"])]
   result = messages[-1]
   return {"messages": messages ,"competitive_set_result": result}