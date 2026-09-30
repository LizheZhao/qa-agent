from agentic_orchestration.diagnostics.contracts import conversation_title


def test_conversation_title_normalizes_whitespace() -> None:
    assert conversation_title("  How does\n incrementality work?  ") == (
        "How does incrementality work?"
    )


def test_conversation_title_is_bounded() -> None:
    title = conversation_title("question " * 40)

    assert len(title) <= 160
    assert title.endswith("…")
