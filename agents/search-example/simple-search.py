# Define a simple graph
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from typing import Annotated, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from tavily import TavilyClient
import os
from PIL import Image
from dotenv import load_dotenv

_ = load_dotenv()

# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    content: List[str]
    output: str

# Add tivaly search tool, and llm
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)

OUTPUT_PROMPT = "Summarize the search results and return an accurate response to the user's query. Start the response with the user's query. Query is {content}"

# Define nodes. Search node, output node
def search_node(state: State):
    query = state["messages"][-1].content
    response = tavily.search(query=query, max_results=5)

    content = []
    for r in response['results']:
        content.append(r['content'])
    return {'content': content}

def output_node(state: State):
    content = "\n\n".join(state['content'] or [])
    response = model.invoke([SystemMessage(OUTPUT_PROMPT.format(content=state['messages'][0].content)), HumanMessage(content)])
    return {'output': response.content}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node('search', search_node)
graph_builder.add_node('response', output_node)
graph_builder.set_entry_point('search')
graph_builder.add_edge('search', 'response')
graph_builder.add_edge('response', END)

graph = graph_builder.compile()
# graph.get_graph().draw_png(output_file_path='./simple-search-graph.png')

thread = {"configurable": {"thread_id": "1"}}
s = graph.invoke({'messages': [HumanMessage('Which is better, Wicked the musical or the movie?')]},thread)
print(s['output'])