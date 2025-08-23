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

from agents.initial_asset_analysis_agent import initial_asset_analysis_agent
from tools.initial_asset_analysis_tools import initial_asset_analysis_tools

from agents.competitive_set_agent import competitive_set_agent
from tools.competitive_set_tools import competitive_set_tools

from agents.detailed_development_analysis_agent import detailed_development_analysis_agent
from tools.detailed_development_analysis_tools import detailed_development_analysis_tools

import streamlit as st

load_dotenv(override=True)


# Define the function that determines whether to continue or not.
# if not go to other agent
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
### FASE 1: ANALISI INIZIALE DELL'ASSET
builder.add_node("initial_asset_assessment", initial_asset_analysis_agent)
builder.add_node("tools_for_initial_asset_assessment", initial_asset_analysis_tools)

### FASE 2: CREAZIONE DEL COMPETITIVE SET (COMPSET)
builder.add_node("competitive_set", competitive_set_agent)
builder.add_node("tools_for_competitive_set", competitive_set_tools)

### FASE 3: ANALISI DETTAGLIATA DEL COMPSET
builder.add_node("detailed_development_analysis", detailed_development_analysis_agent)
builder.add_node("tools_for_detailed_development_analysis", detailed_development_analysis_tools)

#builder.add_node("respond",response_analisi_asset)
# Nodo aggregatore

# Define edges: these determine how the control flow moves
builder.add_edge(START, "initial_asset_assessment")
builder.add_conditional_edges(
    "initial_asset_assessment",
    should_continue,
    {
        "continue": "tools_for_initial_asset_assessment",
        "next_agent": "competitive_set",
    },
)
builder.add_edge("tools_for_initial_asset_assessment", "initial_asset_assessment")

builder.add_conditional_edges(
    "competitive_set",
    should_continue,
    {
        "continue": "tools_for_competitive_set",
        "next_agent": "detailed_development_analysis",
    },
)

builder.add_edge("tools_for_competitive_set", "competitive_set")

builder.add_conditional_edges(
    "detailed_development_analysis",
    should_continue,
    {
        "continue": "tools_for_detailed_development_analysis",
        "next_agent": END,
    },
)

builder.add_edge("tools_for_detailed_development_analysis", "detailed_development_analysis")
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
    #msg_placeholder.write(messages["messages"])
    