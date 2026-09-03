from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.errors import OntologyValidationError
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.bulk_utils import resolve_edge_pointers
from graphiti_core.utils.ontology_utils.strict_ontology import (
    validate_semantic_data_against_ontology,
    validate_strict_ontology_configuration,
)


class Company(BaseModel):
    """A company."""


class Product(BaseModel):
    """A product."""


class Capability(BaseModel):
    """A product capability."""


class Makes(BaseModel):
    """A company makes a product."""


class HasCapability(BaseModel):
    """A product has a capability."""


ENTITY_TYPES = {'Company': Company, 'Product': Product, 'Capability': Capability}
EDGE_TYPES = {'MAKES': Makes, 'HAS_CAPABILITY': HasCapability}
EDGE_TYPE_MAP = {
    ('Company', 'Product'): ['MAKES'],
    ('Product', 'Capability'): ['HAS_CAPABILITY'],
}


def _node(uuid: str, node_type: str) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=uuid,
        group_id='group',
        labels=['Entity', node_type],
    )


def _edge(uuid: str, source: str, target: str, relation: str) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        source_node_uuid=source,
        target_node_uuid=target,
        name=relation,
        group_id='group',
        fact='test fact',
        episodes=['episode'],
        created_at=datetime.now(timezone.utc),
    )


def test_rejects_relation_with_wrong_direction():
    company = _node('company', 'Company')
    product = _node('product', 'Product')
    edge = _edge('edge', product.uuid, company.uuid, 'MAKES')

    with pytest.raises(OntologyValidationError, match='does not allow'):
        validate_semantic_data_against_ontology(
            [company, product], [edge], ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP
        )


def test_rejects_unknown_relation():
    company = _node('company', 'Company')
    product = _node('product', 'Product')
    edge = _edge('edge', company.uuid, product.uuid, 'ACQUIRED')

    with pytest.raises(OntologyValidationError, match='unknown relation type: ACQUIRED'):
        validate_semantic_data_against_ontology(
            [company, product], [edge], ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP
        )


def test_revalidates_relation_after_pointer_resolution():
    company = _node('company', 'Company')
    product = _node('product', 'Product')
    capability = _node('capability', 'Capability')
    edge = _edge('edge', product.uuid, capability.uuid, 'HAS_CAPABILITY')

    validate_semantic_data_against_ontology(
        [product, capability], [edge], ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP
    )

    resolve_edge_pointers([edge], {product.uuid: company.uuid})

    with pytest.raises(OntologyValidationError, match='does not allow'):
        validate_semantic_data_against_ontology(
            [company, capability], [edge], ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP
        )


def test_strict_configuration_must_be_explicit_and_complete():
    with pytest.raises(OntologyValidationError, match='requires'):
        validate_strict_ontology_configuration(ENTITY_TYPES, EDGE_TYPES, None)

    with pytest.raises(OntologyValidationError, match='no allowed signature'):
        validate_strict_ontology_configuration(ENTITY_TYPES, EDGE_TYPES, {})
