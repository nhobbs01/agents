# Entry point for the assistant.
# Input: User request. Reasons about user's input and delegates task to another agent.
# Return: Sub-agent result OR default error when can't handle request with current tools.

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from dotenv import load_dotenv
from TodoManager import TodoListManager
import asyncio

import logfire

logfire.configure()

_ = load_dotenv()


assitant_agent = Agent('gemini-1.5-pro',
                       retries=2,
                        system_prompt="""
                        You are a personal assistant.
                        You task a user's request and decide what action is required to fufull the user's request.
                        You have access to other agents to assist you.
                        You have access to a Todo List Manager to handle any requests relating to a todo list.
                        You can pass the Todo List Manager an instruction using natural language.
                        You cannot assit the user with any other tasks.
                        Respond with the user's name""")

planning_agent = Agent('gemini-1.5-flash',
                       retries=3,
                       result_retries=3,
                        system_prompt="""
                        You are a personal assistant.
                        You task a user's request and decide what action is required to fufull the user's request.
                        Think about the user's request and plan steps to fufill the request.
                        You have acess to the user's Todo List.
                        You don't have any access to any other tools. 
                        Only include instructions for tools you have access to.
                        Keep your instructions consice.
                        Do not prompt the user for more information.                      
                        """)


@assitant_agent.system_prompt
async def add_user_name():
    return "The user's name is Nick."

@assitant_agent.tool
async def manage_todo_list(ctx: RunContext, instruction: str) -> str:
    """Calls the Todo List Manager. The manager can get, update and append to the todo list.

    Args:
        ctx: The context.
        instruction: The instruction to give the Todo List Manager in natural language.
    """
    print('Manage todo list')
    todo = TodoListManager()
    result = await todo.run(instruction=instruction)
    return result    

async def main():
    response = await planning_agent.run('Remind me to call my mum later at 6pm?')
    result = await assitant_agent.run(
        response.data,
    )
    print(result.all_messages())
    print('Response:', result.data)
    print(response.data)


if __name__ == '__main__':
    asyncio.run(main())
