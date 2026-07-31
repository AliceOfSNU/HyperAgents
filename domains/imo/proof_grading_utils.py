QUESTION_ID = "Problem ID"
GROUND_TRUTH_KEY = None  # Ground truth does not exist
MODEL = "openai/o4-mini"  # was internal alias "gpt-o4-mini-genai"; same model, litellm-native routing

def format_input_dict(row):
    # Extract the inputs for the task from the row
    return {
        "domain": "imo_proof_grading",
        "problem": row['Problem'],
        "solution": row['Solution'],
        "grading_guidelines": row['Grading guidelines'],
        "student_answer": row['Response'],
    }