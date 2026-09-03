"""Provider-neutral validation helpers for strict ontology ingestion."""

from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.errors import OntologyValidationError
from graphiti_core.nodes import EntityNode

BASE_ENTITY_TYPE = 'Entity'


def specific_node_types(node: EntityNode) -> frozenset[str]:
    """Return the custom ontology types carried by an entity node."""
    return frozenset(label for label in node.labels if label != BASE_ENTITY_TYPE)


def node_types_match(first: EntityNode, second: EntityNode) -> bool:
    """Return whether two nodes are safe entity-resolution candidates in strict mode."""
    return specific_node_types(first) == specific_node_types(second)


def validate_strict_ontology_configuration(
    entity_types: dict[str, type[BaseModel]] | None,
    edge_types: dict[str, type[BaseModel]] | None,
    edge_type_map: dict[tuple[str, str], list[str]] | None,
) -> None:
    """Require a complete, internally consistent ontology for strict ingestion."""
    if entity_types is None or edge_types is None or edge_type_map is None:
        raise OntologyValidationError(
            'strict_ontology requires entity_types, edge_types, and edge_type_map'
        )

    known_node_types = {BASE_ENTITY_TYPE, *entity_types}
    mapped_edge_types: set[str] = set()
    for (source_type, target_type), relation_types in edge_type_map.items():
        unknown_node_types = {source_type, target_type} - known_node_types
        if unknown_node_types:
            unknown = ', '.join(sorted(unknown_node_types))
            raise OntologyValidationError(
                f'edge_type_map references unknown entity type(s): {unknown}'
            )

        for relation_type in relation_types:
            if relation_type not in edge_types:
                raise OntologyValidationError(
                    f'edge_type_map references unknown relation type: {relation_type}'
                )
            mapped_edge_types.add(relation_type)

    unmapped_edge_types = set(edge_types) - mapped_edge_types
    if unmapped_edge_types:
        unmapped = ', '.join(sorted(unmapped_edge_types))
        raise OntologyValidationError(f'relation type(s) have no allowed signature: {unmapped}')


def validate_nodes_against_ontology(
    nodes: list[EntityNode], entity_types: dict[str, type[BaseModel]]
) -> None:
    """Reject custom node labels that are not declared by the ontology."""
    known_node_types = set(entity_types)
    for node in nodes:
        unknown_node_types = specific_node_types(node) - known_node_types
        if unknown_node_types:
            unknown = ', '.join(sorted(unknown_node_types))
            raise OntologyValidationError(f'node {node.uuid} has unknown entity type(s): {unknown}')


def validate_edges_against_ontology(
    edges: list[EntityEdge],
    nodes: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
) -> None:
    """Reject unknown relations and relations whose endpoint types violate their signature."""
    nodes_by_uuid = {node.uuid: node for node in nodes}
    mapped_relation_types = {
        relation_type
        for relation_types in edge_type_map.values()
        for relation_type in relation_types
    }

    for edge in edges:
        if edge.name not in edge_types or edge.name not in mapped_relation_types:
            raise OntologyValidationError(
                f'edge {edge.uuid} has unknown relation type: {edge.name}'
            )

        source_node = nodes_by_uuid.get(edge.source_node_uuid)
        target_node = nodes_by_uuid.get(edge.target_node_uuid)
        if source_node is None or target_node is None:
            raise OntologyValidationError(
                f'edge {edge.uuid} references an entity outside the validated node set'
            )

        source_types = {BASE_ENTITY_TYPE, *specific_node_types(source_node)}
        target_types = {BASE_ENTITY_TYPE, *specific_node_types(target_node)}
        allowed = any(
            edge.name in relation_types
            and source_type in source_types
            and target_type in target_types
            for (source_type, target_type), relation_types in edge_type_map.items()
        )
        if not allowed:
            source = ', '.join(sorted(source_types))
            target = ', '.join(sorted(target_types))
            raise OntologyValidationError(
                f'edge {edge.uuid} relation {edge.name} does not allow '
                f'source type(s) [{source}] and target type(s) [{target}]'
            )


def validate_semantic_data_against_ontology(
    nodes: list[EntityNode],
    edges: list[EntityEdge],
    entity_types: dict[str, type[BaseModel]],
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
) -> None:
    """Validate a complete semantic write set against a declared ontology."""
    validate_nodes_against_ontology(nodes, entity_types)
    validate_edges_against_ontology(edges, nodes, edge_types, edge_type_map)
