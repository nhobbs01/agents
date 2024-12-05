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
    content: List[str] = []
    task: str
    plan: str
    queries: List[str]
    critique: str
    revision_num: int
    max_revisions: int
    output: str

from langchain_core.pydantic_v1 import BaseModel

class Queries(BaseModel):
    queries: List[str]

# Add tivaly search tool, and llm
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)

PLAN_PROMPT = "You are a Personal assistant that specializes in searching the web for answers to a user's query. Think about the user's query and think to yourself about what the user means by their question and different angles you could use to approach this question."
GENERATE_QUERIES_PROMPT = "You are a search assistant tasked with searching the web for answers. Generate web search queries for the user's request, using all information from the plan. This is to gather information to help in giving a user an answer to their query {plan}. Here is what the user already knows and they would like to delve deeper into the question: {draft}. Here is a critique of the users current knowledge: {critique}"
GENERATE_DRAFT_PROMPT = "Summarize the search results and return an accurate response to the user's query. Start the response with the user's query. Query is {content}"
REFLECT_NODE_PROMPT = "You are a tasked with judging the quality of an output to a query. Critique the response to the query and provide helpful feedback that can guide a user to continue their search for the answer. {draft}"
# Define nodes. Plan Node, Generate queries, Search node, Generate Draft, Reflect Node
def search_node(state: State):
    content = state['content'] or []
    for q in state['queries']:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {'content': content}

def plan_node(state: State):
    plan = model.invoke([SystemMessage(PLAN_PROMPT), HumanMessage(state['task'])])
    return {'plan': plan.content}

def generate_queries(state: State):
    queries = model.with_structured_output(Queries).invoke([SystemMessage(GENERATE_QUERIES_PROMPT.format(plan=state['plan'],
                                 draft=state['output'], critique=state['critique'])), HumanMessage(state['task'])])
    return {'queries': queries.queries}

def reflect_node(state: State):
    critique = model.invoke([SystemMessage(REFLECT_NODE_PROMPT.format(draft=state['output'])), HumanMessage(state['task'])])
    return {'revision_num': state['revision_num'] + 1, 'critique': critique.content}

def generate_draft(state: State):
    content = "\n\n".join(state['content'] or [])
    response = model.invoke([SystemMessage(GENERATE_DRAFT_PROMPT.format(content=state['task'])), HumanMessage(content)])
    return {'output': response.content}

def should_continue(state: State):
    if(state['revision_num'] >= state['max_revisions']):
        return END
    return 'reflect' 


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node('plan_node', plan_node)
graph_builder.add_node('gen_queries', generate_queries)
graph_builder.add_node('search', search_node)
graph_builder.add_node('generate_draft', generate_draft)
graph_builder.add_node('reflect', reflect_node)

graph_builder.set_entry_point('plan_node')
graph_builder.add_edge('plan_node','gen_queries')
graph_builder.add_edge('gen_queries','search')
graph_builder.add_edge('search','generate_draft')
graph_builder.add_conditional_edges('generate_draft',should_continue, {END: END, "reflect": "reflect"} )
graph_builder.add_edge('reflect', 'gen_queries')
graph = graph_builder.compile()
# graph.get_graph().draw_png(output_file_path='./simple-search-graph.png')

thread = {"configurable": {"thread_id": "1"}}
result = graph.invoke({'task': 'Which is better, Wicked the musical or the movie?', "max_revisions": 2,
    "revision_num": 1, 'output':'', 'critique':'', 'content': []},thread)

print(result['output'])
