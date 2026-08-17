from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from utils.common import extract_jsons

class TaskAgent(AgentSystem):
    def forward(self, inputs):
        """
        An agent that solves a given task.

        Args:
            inputs (dict): A dictionary with input data for the task.

        Returns:
            tuple:
                - prediction (str): The prediction made by the agent.
                - new_msg_history (list): A list of messages representing the message history of the interaction.
        """
        domain = inputs['domain']
        repo_path = inputs.get('git_tempdir', '.')  # The repository the agent should modify
        instruction = f"""You are an agent that solves a coding task by directly modifying the repository at `{repo_path}`.

Task input:
```
{inputs}
```

You have access to a bash shell and an editor. Follow this process:
1. Explore the repository at `{repo_path}` (e.g. `view` files, `bash` to list and inspect).
2. Identify what code changes are needed to solve the problem.
3. Make focused edits to the repository using the `editor` tool (or `bash` for commands like running tests).
4. Iterate until you are confident the solution works.
5. Finally, respond in JSON format with the following schema (and nothing else after it):
<json>
{{
    "response": "a concise summary of the changes you made"
}}
</json>

Do NOT attempt to create the patch file yourself; the harness will compute the diff from the repository after you finish."""
        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
        )

        # Extract the response
        prediction = "None"
        try:
            extracted_jsons = extract_jsons(new_msg_history[-1]['text'])
            if extracted_jsons is not None and "response" in extracted_jsons[-1]:
                prediction = extracted_jsons[-1]['response']
        except Exception as e:
            self.log(f"Error extracting prediction: {e}")
            prediction = "None"

        return prediction, new_msg_history
