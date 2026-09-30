import pytest

from agentic_orchestration.diagnostics.mongodb import _turn_summary_pipeline


@pytest.mark.unit
def test_turn_summary_query_projects_only_the_final_generated_message() -> None:
    pipeline = _turn_summary_pipeline({"session_id": "session"}, 25)

    assert pipeline[:3] == [
        {"$match": {"session_id": "session"}},
        {"$sort": {"turn_number": -1}},
        {"$limit": 26},
    ]
    assert pipeline[3]["$project"]["generated_messages"] == {
        "$filter": {
            "input": "$generated_messages",
            "as": "message",
            "cond": {"$eq": ["$$message.id", "$final_message_id"]},
        }
    }
