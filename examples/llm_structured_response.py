from src.integrations.llm import generate_structured_response


if __name__ == "__main__":
    query = "How did overall hispanic tv perform comparing 2023 to 2022?"
    valid_options = ["fiscal year 2020", "fiscal year 2021", "fiscal year 2022", "fiscal year 2023"]

    # Generate Free text on time variable
    response_structure_1 = {
        "time": {
            "type": "string",
        }
    }

    # Generate select 1 from valid options on time variable
    response_structure_2 = {
        "time": {
            "type": "string",
            "enum": valid_options
        }
    }

    # Generate select multiple from valid options on time variable
    response_structure_3 = {
        "time": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": valid_options
            }
        }
    }

    print(generate_structured_response(query, response_structure_1))
    print(generate_structured_response(query, response_structure_2))
    print(generate_structured_response(query, response_structure_3))
