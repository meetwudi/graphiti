from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import OntologyValidationError
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode


class Company(BaseModel):
    """A company."""


class Product(BaseModel):
    """A product."""


class Makes(BaseModel):
    """A company makes a product."""


ENTITY_TYPES = {'Company': Company, 'Product': Product}
EDGE_TYPES = {'MAKES': Makes}
EDGE_TYPE_MAP = {('Company', 'Product'): ['MAKES']}


def _graphiti() -> Graphiti:
    graphiti = Graphiti.__new__(Graphiti)
    driver = MagicMock()
    driver._database = 'group'
    driver.provider = GraphProvider.NEO4J
    graphiti.driver = driver
    graphiti.clients = SimpleNamespace(driver=driver)
    graphiti.embedder = MagicMock()
    graphiti.store_raw_episode_content = True
    graphiti.max_coroutines = None

    span = MagicMock()
    span_context = MagicMock()
    span_context.__enter__.return_value = span
    span_context.__exit__.return_value = False
    graphiti.tracer = MagicMock()
    graphiti.tracer.start_span.return_value = span_context
    return graphiti


def _node(uuid: str, node_type: str) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=uuid,
        group_id='group',
        labels=['Entity', node_type],
    )


def _edge(uuid: str, source: str, target: str) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        source_node_uuid=source,
        target_node_uuid=target,
        name='MAKES',
        group_id='group',
        fact='test fact',
        episodes=['episode'],
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_single_post_resolution_failure_prevents_persistence(monkeypatch):
    graphiti = _graphiti()
    company = _node('company', 'Company')
    product = _node('product', 'Product')
    invalid_edge = _edge('invalid', product.uuid, company.uuid)

    graphiti.retrieve_episodes = AsyncMock(return_value=[])
    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_nodes',
        AsyncMock(return_value=([company, product], {})),
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_nodes',
        AsyncMock(
            return_value=(
                [company, product],
                {company.uuid: company.uuid, product.uuid: product.uuid},
                [],
            )
        ),
    )
    graphiti._extract_and_resolve_edges = AsyncMock(
        return_value=([invalid_edge], [], [invalid_edge])
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_attributes_from_nodes',
        AsyncMock(return_value=[company, product]),
    )
    graphiti._process_episode_data = AsyncMock()

    with pytest.raises(OntologyValidationError, match='does not allow'):
        await graphiti.add_episode(
            name='episode',
            episode_body='content',
            source_description='test',
            reference_time=datetime.now(timezone.utc),
            group_id='group',
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
            strict_ontology=True,
        )

    graphiti._process_episode_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_missing_raw_edge_endpoint_prevents_persistence(monkeypatch):
    graphiti = _graphiti()
    company = _node('company', 'Company')
    product = _node('product', 'Product')
    graphiti.clients.llm_client = MagicMock()
    graphiti.clients.llm_client.generate_response = AsyncMock(
        return_value={
            'edges': [
                {
                    'source_entity_name': company.name,
                    'target_entity_name': 'missing-product',
                    'relation_type': 'MAKES',
                    'fact': 'The company makes the missing product.',
                }
            ]
        }
    )

    graphiti.retrieve_episodes = AsyncMock(return_value=[])
    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_nodes',
        AsyncMock(return_value=([company, product], {})),
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_nodes',
        AsyncMock(
            return_value=(
                [company, product],
                {company.uuid: company.uuid, product.uuid: product.uuid},
                [],
            )
        ),
    )
    graphiti._process_episode_data = AsyncMock()

    with pytest.raises(OntologyValidationError, match='Target entity.*not found'):
        await graphiti.add_episode(
            name='episode',
            episode_body='content',
            source_description='test',
            reference_time=datetime.now(timezone.utc),
            group_id='group',
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
            strict_ontology=True,
        )

    graphiti._process_episode_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_invalid_item_aborts_all_semantic_writes(monkeypatch):
    graphiti = _graphiti()
    company = _node('company', 'Company')
    product = _node('product', 'Product')
    valid_edge = _edge('valid', company.uuid, product.uuid)
    invalid_edge = _edge('invalid', product.uuid, company.uuid)

    add_bulk = AsyncMock(return_value=None)
    monkeypatch.setattr('graphiti_core.graphiti.add_nodes_and_edges_bulk', add_bulk)
    monkeypatch.setattr(
        'graphiti_core.graphiti.retrieve_previous_episodes_bulk',
        AsyncMock(side_effect=lambda _, e: [(episode, []) for episode in e]),
    )

    async def extract(episode_context, *_args):
        return (
            {episode.uuid: [company, product] for episode, _ in episode_context},
            {},
            [[] for _ in episode_context],
        )

    graphiti._extract_and_dedupe_nodes_bulk = AsyncMock(side_effect=extract)
    monkeypatch.setattr(
        'graphiti_core.graphiti.dedupe_edges_bulk',
        AsyncMock(
            side_effect=lambda _clients, _edges, contexts, *_args: {
                episode.uuid: [] for episode, _ in contexts
            }
        ),
    )
    graphiti._resolve_nodes_and_edges_bulk = AsyncMock(
        return_value=([company, product], [valid_edge, invalid_edge], [], {})
    )

    episodes = [
        RawEpisode(
            name=f'episode-{index}',
            content='content',
            source_description='test',
            source=EpisodeType.message,
            reference_time=datetime.now(timezone.utc),
        )
        for index in range(2)
    ]

    with pytest.raises(OntologyValidationError, match='does not allow'):
        await graphiti.add_episode_bulk(
            episodes,
            group_id='group',
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
            strict_ontology=True,
        )

    add_bulk.assert_awaited_once()
    registration = add_bulk.await_args.kwargs
    assert len(registration['episodic_nodes']) == 2
    assert registration['entity_nodes'] == []
    assert registration['entity_edges'] == []
    assert registration['episodic_edges'] == []


@pytest.mark.asyncio
async def test_bulk_raw_invalid_edge_stops_resolution_and_semantic_persistence(monkeypatch):
    graphiti = _graphiti()
    company = _node('company', 'Company')
    product = _node('product', 'Product')
    wrong_direction = _edge('wrong-direction', product.uuid, company.uuid)

    add_bulk = AsyncMock(return_value=None)
    monkeypatch.setattr('graphiti_core.graphiti.add_nodes_and_edges_bulk', add_bulk)
    monkeypatch.setattr(
        'graphiti_core.graphiti.retrieve_previous_episodes_bulk',
        AsyncMock(side_effect=lambda _, episodes: [(episode, []) for episode in episodes]),
    )
    monkeypatch.setattr(
        'graphiti_core.utils.bulk_utils.extract_nodes',
        AsyncMock(return_value=([company, product], {})),
    )
    monkeypatch.setattr(
        'graphiti_core.utils.bulk_utils.extract_edges',
        AsyncMock(return_value=[wrong_direction]),
    )
    resolve_bulk = AsyncMock()
    monkeypatch.setattr('graphiti_core.utils.bulk_utils.dedupe_nodes_bulk', resolve_bulk)
    graphiti._resolve_nodes_and_edges_bulk = AsyncMock()

    episode = RawEpisode(
        name='episode',
        content='content',
        source_description='test',
        source=EpisodeType.message,
        reference_time=datetime.now(timezone.utc),
    )

    with pytest.raises(OntologyValidationError, match='does not allow'):
        await graphiti.add_episode_bulk(
            [episode],
            group_id='group',
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
            strict_ontology=True,
        )

    resolve_bulk.assert_not_awaited()
    graphiti._resolve_nodes_and_edges_bulk.assert_not_awaited()
    add_bulk.assert_awaited_once()
    registration = add_bulk.await_args.kwargs
    assert len(registration['episodic_nodes']) == 1
    assert registration['entity_nodes'] == []
    assert registration['entity_edges'] == []
    assert registration['episodic_edges'] == []


@pytest.mark.asyncio
async def test_bulk_missing_raw_edge_endpoint_stops_semantic_persistence(monkeypatch):
    graphiti = _graphiti()
    company = _node('company', 'Company')
    product = _node('product', 'Product')
    graphiti.clients.llm_client = MagicMock()
    graphiti.clients.llm_client.generate_response = AsyncMock(
        return_value={
            'edges': [
                {
                    'source_entity_name': 'missing-company',
                    'target_entity_name': product.name,
                    'relation_type': 'MAKES',
                    'fact': 'The missing company makes the product.',
                }
            ]
        }
    )

    add_bulk = AsyncMock(return_value=None)
    monkeypatch.setattr('graphiti_core.graphiti.add_nodes_and_edges_bulk', add_bulk)
    monkeypatch.setattr(
        'graphiti_core.graphiti.retrieve_previous_episodes_bulk',
        AsyncMock(side_effect=lambda _, episodes: [(episode, []) for episode in episodes]),
    )
    monkeypatch.setattr(
        'graphiti_core.utils.bulk_utils.extract_nodes',
        AsyncMock(return_value=([company, product], {})),
    )
    dedupe_nodes = AsyncMock()
    monkeypatch.setattr('graphiti_core.utils.bulk_utils.dedupe_nodes_bulk', dedupe_nodes)
    graphiti._resolve_nodes_and_edges_bulk = AsyncMock()

    episode = RawEpisode(
        name='episode',
        content='content',
        source_description='test',
        source=EpisodeType.message,
        reference_time=datetime.now(timezone.utc),
    )

    with pytest.raises(OntologyValidationError, match='Source entity.*not found'):
        await graphiti.add_episode_bulk(
            [episode],
            group_id='group',
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
            strict_ontology=True,
        )

    prompt_messages = graphiti.clients.llm_client.generate_response.await_args.args[0]
    prompt_text = '\n'.join(message.content for message in prompt_messages)
    assert 'FACT_TYPES are exhaustive' in prompt_text
    assert 'Never invent a relation_type' in prompt_text
    dedupe_nodes.assert_not_awaited()
    graphiti._resolve_nodes_and_edges_bulk.assert_not_awaited()
    add_bulk.assert_awaited_once()
    registration = add_bulk.await_args.kwargs
    assert len(registration['episodic_nodes']) == 1
    assert registration['entity_nodes'] == []
    assert registration['entity_edges'] == []
    assert registration['episodic_edges'] == []
