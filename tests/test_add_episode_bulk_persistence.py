from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import graphiti_core.graphiti as graphiti_module
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.graphiti import Graphiti
from graphiti_core.models.nodes.node_db_queries import get_episode_node_save_bulk_query
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.bulk_utils import RawEpisode


class SpanStub:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def add_attributes(self, attributes):
        pass

    def set_status(self, status, description=None):
        pass

    def record_exception(self, exception):
        pass


class TracerStub:
    def start_span(self, name):
        return SpanStub()


@pytest.mark.asyncio
async def test_add_episode_bulk_does_not_persist_raw_episodes_before_semantic_processing(
    monkeypatch,
) -> None:
    driver = SimpleNamespace(provider=GraphProvider.NEO4J, _database='publisher-graph')
    graphiti: Any = object.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.clients = SimpleNamespace(driver=driver)
    graphiti.embedder = SimpleNamespace()
    graphiti.tracer = TracerStub()
    graphiti._extract_and_dedupe_nodes_bulk = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError('semantic extraction failed')
    )

    persist = AsyncMock()
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', persist)
    retrieve_previous = AsyncMock(return_value=[])
    monkeypatch.setattr(
        graphiti_module,
        'retrieve_previous_episodes_bulk',
        retrieve_previous,
    )
    monkeypatch.setattr(
        EpisodicNode,
        'get_by_uuid',
        AsyncMock(side_effect=NodeNotFoundError('deterministic-episode')),
    )

    episode = RawEpisode(
        uuid='deterministic-episode',
        name='First',
        content='First article',
        source_description='test source',
        source=EpisodeType.message,
        reference_time=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )

    with pytest.raises(RuntimeError, match='semantic extraction failed'):
        await graphiti.add_episode_bulk([episode], group_id='publisher-graph')

    persist.assert_not_awaited()
    prepared_episodes = retrieve_previous.await_args.args[1]
    assert [prepared.uuid for prepared in prepared_episodes] == ['deterministic-episode']


@pytest.mark.parametrize(
    'provider',
    [GraphProvider.NEPTUNE, GraphProvider.FALKORDB, GraphProvider.NEO4J],
)
def test_bulk_episode_upsert_preserves_completion_properties(provider: GraphProvider) -> None:
    query = get_episode_node_save_bulk_query(provider)
    assert 'SET n +=' in query
    assert 'SET n =' not in query
