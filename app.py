# Import libraries
import os
import yaml
from dotenv import load_dotenv

from langgraph.graph import StateGraph,  START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

from IPython.display import Image, display

# Import agent packages
from utils.states import AgentState
from tools.analisi_asset import *
from tools.analisi_compset import *
from tools.compset import *

import streamlit as st

load_dotenv(override=True)


# Create model as brain
model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_MODEL"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))


# Create tools
tools_for_analisi_asset = [get_coordinates, calculate_distance_to_city_centers, get_distance_between_coordinates, download_satellite_image, estimate_scale, calculate_area]
tools_for_compset = [get_coordinates]
tools_for_analisi_compset = [get_coordinates]


model_with_tools_for_compset         = model.bind_tools(tools_for_compset, parallel_tool_calls=False)
model_with_tools_for_analisi_compset = model.bind_tools(tools_for_analisi_compset, parallel_tool_calls=False)

# Import prompts

sys_msg_compset = SystemMessage(content=""" """)
sys_msg_analisi_compset = SystemMessage(content=""" """)

# Define Agents
######################## ANALISI ASSET AGENT ###################################################


######################## COMPSET AGENT #########################################################
def compset_agent(state: AgentState):
   messages = [model_with_tools_for_compset.invoke([sys_msg_compset] + state["messages"])]
   compset_result = messages[-1]
   return {"messages": messages, "compset_result": compset_result}

######################## ANALISI COMPSET AGENT #################################################
def analisi_compset_agent(state: AgentState):
   # Construct the final answer from the arguments of the last call
   messages = [model_with_tools_for_analisi_compset.invoke([sys_msg_analisi_compset] + state["messages"])]
   analisi_compset_result = messages[-1]
   return {"messages": messages, "analisi_compset_result": analisi_compset_result}



# Define the function that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is only one tool call and it is the response tool call we respond to the user
    if last_message.tool_calls:
        return "continue"
    # Otherwise we will use the tool node again
    else:
        return "next_agent"

# Initialize Graph
builder = StateGraph(AgentState)

# Define nodes: these do the work for the Agent 
builder.add_node("analisi_asset", analisi_asset_agent)
builder.add_node("tools_for_analisi_asset", ToolNode(tools_for_analisi_asset))

builder.add_node("compset", compset_agent)
builder.add_node("tools_for_compset", ToolNode(tools_for_compset))

builder.add_node("analisi_compset", analisi_compset_agent)
builder.add_node("tools_for_analisi_compset", ToolNode(tools_for_analisi_compset))

#builder.add_node("respond",response_analisi_asset)

# Define edges: these determine how the control flow moves
builder.add_edge(START, "analisi_asset")
builder.add_conditional_edges(
    "analisi_asset",
    should_continue,
    {
        "continue": "tools_for_analisi_asset",
        "next_agent": "compset",
    },
)
builder.add_edge("tools_for_analisi_asset", "analisi_asset")

builder.add_conditional_edges(
    "compset",
    should_continue,
    {
        "continue": "tools_for_compset",
        "next_agent": "analisi_compset",
    },
)

builder.add_edge("tools_for_compset", "compset")

builder.add_conditional_edges(
    "analisi_compset",
    should_continue,
    {
        "continue": "tools_for_analisi_compset",
        "next_agent": END,
    },
)

builder.add_edge("tools_for_analisi_compset", "analisi_compset")
#builder.add_edge("respond", END)

react_graph = builder.compile()


# Save image
#image = Image(react_graph.get_graph(xray=True).draw_mermaid_png())



# Run Agent
#messages = [HumanMessage(content="You need to analyze the hotel Les Terraces d'Eze")]
#messages = react_graph.invoke({"messages": messages}, interrupt_before="analisi_compset")
#for m in messages['messages']:
#    m.pretty_print()

if "messages" not in st.session_state:
    # default initial message to render in message state
    st.session_state["messages"] = [AIMessage(content="How can I help you?")]

for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

with st.chat_message("assistant"):
    msg_placeholder = st.empty() 
    # create a new placeholder for streaming messages and other events, and give it context

    messages = react_graph.invoke({"messages": st.session_state.messages}, interrupt_before="analisi_compset", config={"recursion_limit": 1000})
    last_msg = messages["messages"][-1].content
    st.session_state.messages.append(AIMessage(content=last_msg))  # Add that last message to the st_message_state
    msg_placeholder.write(messages["messages"])