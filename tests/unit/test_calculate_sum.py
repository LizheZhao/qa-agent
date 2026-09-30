import pytest
from orchestration_tools.arithmetic import calculate_sum, calculate_sum_tool
from pydantic import ValidationError


@pytest.mark.unit
def test_calculate_sum_validates_and_adds_numbers() -> None:
    assert calculate_sum(17, 25) == 42.0
    with pytest.raises(ValidationError):
        calculate_sum_tool.invoke({"a": "not-a-number", "b": 2})
    with pytest.raises(ValidationError):
        calculate_sum_tool.invoke({"a": 1, "b": 2, "unexpected": 3})
