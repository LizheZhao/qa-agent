"""Normalize compiled LangGraph topology into versioned graph definitions."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from orchestration_core import (
    NODE_KIND_METADATA_KEY,
    PROMPT_TEMPLATE_ID_METADATA_KEY,
    PROMPT_TEMPLATE_METADATA_KEY,
    GraphDefinition,
    GraphEdgeDefinition,
    GraphEdgeKind,
    GraphNodeDefinition,
    GraphNodeKind,
    PromptTemplateDefinition,
)


def build_graph_definition(
    graph: Any,
    *,
    agent_id: str,
    agent_version: str,
    entrypoint: str,
) -> GraphDefinition:
    """Extract public topology without depending on private LangGraph structures."""

    drawable = graph.get_graph()
    nodes = tuple(
        _node_definition(node_id, node) for node_id, node in sorted(drawable.nodes.items())
    )
    edges = tuple(
        GraphEdgeDefinition(
            source_node_id=edge.source,
            target_node_id=edge.target,
            kind=GraphEdgeKind.CONDITIONAL if edge.conditional else GraphEdgeKind.DIRECT,
            label=(str(edge.data) if edge.data is not None else edge.target)
            if edge.conditional
            else None,
        )
        for edge in sorted(
            drawable.edges,
            key=lambda item: (
                item.source,
                item.target,
                item.conditional,
                str(item.data or ""),
            ),
        )
    )
    topology = {
        "entrypoint": entrypoint,
        "nodes": [node.model_dump(mode="json") for node in nodes],
        "edges": [edge.model_dump(mode="json") for edge in edges],
    }
    digest = sha256(
        json.dumps(topology, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return GraphDefinition(
        definition_id=f"{agent_id}:{agent_version}:{digest[:16]}",
        definition_hash=digest,
        agent_id=agent_id,
        agent_version=agent_version,
        entrypoint=entrypoint,
        nodes=nodes,
        edges=edges,
        created_at=datetime.now(UTC),
    )


def graph_predecessors(definition: GraphDefinition) -> Mapping[str, tuple[str, ...]]:
    """Index possible runtime predecessors from one immutable graph definition."""

    predecessors: defaultdict[str, list[str]] = defaultdict(list)
    for edge in definition.edges:
        if edge.source_node_id != "__start__" and edge.target_node_id != "__end__":
            predecessors[edge.target_node_id].append(edge.source_node_id)
    return {node_id: tuple(sorted(source_ids)) for node_id, source_ids in predecessors.items()}


def _node_definition(node_id: str, node: Any) -> GraphNodeDefinition:
    metadata = node.metadata or {}
    if node_id == "__start__":
        kind = GraphNodeKind.START
    elif node_id == "__end__":
        kind = GraphNodeKind.END
    else:
        kind = GraphNodeKind(metadata.get(NODE_KIND_METADATA_KEY, "deterministic"))
    prompt_id = metadata.get(PROMPT_TEMPLATE_ID_METADATA_KEY)
    prompt_text = metadata.get(PROMPT_TEMPLATE_METADATA_KEY)
    prompt = (
        PromptTemplateDefinition(template_id=prompt_id, template_text=prompt_text)
        if isinstance(prompt_id, str) and isinstance(prompt_text, str)
        else None
    )
    return GraphNodeDefinition(
        node_id=node_id,
        display_name=node.name,
        kind=kind,
        prompt_template=prompt,
    )
