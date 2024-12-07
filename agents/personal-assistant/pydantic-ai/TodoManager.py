from pydantic_ai import Agent, Tool

class TodoListManager:

    def __init__(self):
        self.agent = Agent('gemini-1.5-flash',
                         tools=[Tool(self.append_todo_list, takes_ctx=False),Tool(self.get_todo_list, takes_ctx=False), Tool(self.update_todo_list, takes_ctx=False)])

    async def run(self, instruction: str) -> str:
        print('Running todo list manager: ',instruction )
        result = await self.agent.run(instruction)
        print(result.data)
        return result.data
        

    async def append_todo_list(self, item: str):
        print('Appending to todo list', item)
        return 'Success'
    
    async def update_todo_list(self, item: str):
        print('Appending to todo list', item)
        return 'Success'
    
    async def get_todo_list(self):
        print('Getting todo list')
        return "Todo: Code assistant agent."