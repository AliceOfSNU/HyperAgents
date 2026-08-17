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
        problem_statement = inputs.get('problem_statement', '')
        git_tempdir = inputs.get('git_tempdir', '')
        base_commit = inputs.get('base_commit', '')
        test_description = inputs.get('test_description', '')
        language = inputs.get('language', '')

        instruction = f"""You are an autonomous coding agent. Your task is to solve the given problem by modifying the codebase in `{git_tempdir}`.

Problem statement:
{problem_statement}

Base commit: {base_commit}
Language: {language or 'unknown'}
Test description (if any): {test_description or 'none'}

You have access to bash and file editing tools. Work in the provided repository to implement a solution. Make the necessary code changes, run tests if possible, and ensure your changes are complete.

After you finish making changes, respond in JSON format with the following schema:
<json>
{{
    "response": "A brief summary of the changes you made and the final prediction/answer if applicable."
}}
</json>"""
        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
            multiple_tool_calls=True,
            max_tool_calls=50,
        )

        # Extract the response (look for the last JSON with 'response' key)
        prediction = "None"
        try:
            for msg in reversed(new_msg_history):
                if 'text' in msg:
                    extracted_jsons = extract_jsons(msg['text'])
                    if extracted_jsons:
                        for extracted in reversed(extracted_jsons):
                            if isinstance(extracted, dict) and "response" in extracted:
                                prediction = extracted['response']
                                break
                    if prediction != "None":
                        break
        except Exception as e:
            self.log(f"Error extracting prediction: {e}")
            prediction = "None"

        return prediction, new_msg_history
