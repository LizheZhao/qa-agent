custom_field_base_prompt = """
You are a natural language processing engine, specialized in marketing data analysis.
You are asked to perform a classification of whether the input question is relevant to {cols} following closely the question and examples given. Do not imply or assume additional information.
    Classify the input question to one of the following:
    A. irrelevant: The question does not mention {cols} at all.
    B. all: The question refers to {cols} in general, usually using words like by, each, for all, different, or all other followed by {cols}.
    Also select all if the question compares one specific {cols} to the rest or other {cols}.
    C. specific: The question discusses specific {cols} names that appear in the {predefined_list}.
    
    # input question
    {question}
    
    # context
    {context}

    # here are some examples:
    {example}

    Output the classification result in JSON format with the following fields only: {cols} with the following structure:
    {answer_format}
    Do not explain. Do not include additional content. Do not imply from question and context
"""