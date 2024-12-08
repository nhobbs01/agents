from pydantic_ai import Agent, Tool

class TodoListManager:

    def __init__(self):
        self.agent = Agent('gemini-1.5-flash',
              system_prompt="""
                        You are a todo list manager. You help the user manage their todo list.
                        You can fetch, update and append the user's todo list.
                        First fetch the todo list, then analzye the user's request.
                        Create a plan for the steps required to help the user. Fetch the todo list fisrt to help you plan.
                        Don't append to the list if the user's request is already on the todo list. 
                        """,
                         tools=[Tool(self.get_todo_list, takes_ctx=False),Tool(self.append_todo_list, takes_ctx=False), Tool(self.update_todo_list, takes_ctx=False)])

    async def run(self, instruction: str) -> str:
        print('Running todo list manager: ',instruction )     
        result = await self.agent.run(instruction)
        return result.data
        

    async def append_todo_list(self, item: str) -> str:
        """Appends an item to the todo list."""
        print('Appending', item)
        with open("test.txt", "a") as myfile:
            myfile.write(item + '\n')

        return 'Success'

    async def update_todo_list(self, item: str) -> str:
        """Will overwrite any existing content"""
        print('Updating to todo list', item)
        with open("test.txt", "w") as myfile:
            myfile.write(item + '\n')
        return 'Success'

    async def get_todo_list(self) -> str:
        """Returns the current todo list"""
        print('Getting todo list')
        res = ""
        with open("test.txt", "r") as myfile:
            res = myfile.read()
        return res