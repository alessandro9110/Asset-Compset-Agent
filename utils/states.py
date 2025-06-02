from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages, AnyMessage

class AgentState(TypedDict):
    """The state of the agent."""
    # SUperificie totalde dell'area in metri quadri
    messages: Annotated[list[AnyMessage], add_messages]
    
    analisi_asset_result: str
    compset_result: str
    analisi_compset_result: str