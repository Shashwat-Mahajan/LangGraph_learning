from dotenv import load_dotenv
load_dotenv()
import os
from typing import Annotated
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt,Command

class State(TypedDict):
    messages:Annotated[list,add_messages]


def chatbot(state: State) -> State:
    return {'messages':[llm.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)

builder.add_edge(START,"chatbot_node")
builder.add_edge("chatbot_node",END)

graph = builder.compile()

@tool
def get_stock_price(symbol: str) -> float:
    '''Return the current price of a stock given the stock symbol
    :param symbol:stock symbol
    :return: current price of the stock
    '''
    return {
        "MSFT": 200.3,
        "AAPL": 100.4,
        "AMZN" : 150.0,
        "RIL":87.6
    }.get(symbol, 0.0)

@tool
def buy_stocks(symbol:str,quantity:int,total_price:float) -> str:
    """
To buy stocks:
1. First call get_stock_price.
2. Wait for the result.
3. Then call buy_stocks.
Never place one tool call inside another tool's arguments.
"""

    decision = interrupt(f"Approve buying {quantity} {symbol} stocks for ${total_price:.2f}")

    if decision == "yes":
        return f"You bought {quantity} shares of {symbol} stock at a total price of {total_price}"
    else:
        return "Buying declined."


tools = [get_stock_price, buy_stocks]
llm = init_chat_model("openai/gpt-oss-120b",
    model_provider="groq")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {'messages':[llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node(chatbot)
builder.add_node("tools",ToolNode(tools))
builder.add_edge(START,"chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools","chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=memory)

from IPython.display import Image,display
display(Image(graph.get_graph().draw_mermaid_png()))

config = {'configurable':{'thread_id':'1'}}

#State 1: user asks price
# state = graph.invoke({"messages":[{"role":"user","content":"i want to buy 20 AMZN stock using the current price and 5 MSFT stock, what will be the total cost?"}]},config=config)


#Step 2: user asks to buy
state = graph.invoke({"messages":[{"role":"user","content":"Buy 10 MSFT stocks at current price."}]},config=config)
print(state.get("__interrupt__"))

decision = input("Approve (yes/no): ")
state = graph.invoke(Command(resume=decision), config=config)
print(state["messages"][-1].content)


