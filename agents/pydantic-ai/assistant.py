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


assitant_agent = Agent('gemini-1.5-flash',
                       retries=2,
                        system_prompt="""
                        You are a personal assistant.
                        You task a user's request and decide what action is required to fufull the user's request.
                        You have access to other agents to assist you.
                        You have access to a Todo List Manager to handle any requests relating to a todo list.
                        Listen to the response from the Todo List Manager before answering back to the user.
                       """)

response_agent = Agent('gemini-1.5-flash',
                        system_prompt="""
                        You are a personal assistant.
                        Review the steps and craft a response to the user from the tool actions taken.
                        Think about the result from the Todo List Manager. 
                        Use the result from the Todo list manager to guide your response.
                        Explain to the user what action the agent has taken.
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
    response = await assitant_agent.run('I need to take the bins out then clean the kitchen?')
    print(response.all_messages())
    print('Response:', response.data)


if __name__ == '__main__':
    asyncio.run(main())
