from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast

import pytest
from graphiti_core.driver.driver import GraphDriver, GraphProvider

from graph_service.zep_graphiti import ZepGraphiti


@pytest.mark.asyncio
async def test_episode_ingest_state_uses_exact_episode_uuids_and_completion_marker() -> None:
    valid_at = datetime(2026, 9, 2, tzinfo=timezone.utc)

    class DriverStub:
        provider = GraphProvider.NEO4J

        def __init__(self):
            self.query = ''
            self.parameters = {}

        async def execute_query(self, query, **parameters):
            self.query = query
            self.parameters = parameters
            return (
                [
                    {
                        'uuid': 'episode-1',
                        'group_id': 'another-graph',
                        'name': 'First',
                        'content': 'source(system): First',
                        'source_description': 'first source',
                        'source': 'message',
                        'valid_at': valid_at,
                        'completed': True,
                    }
                ],
                None,
                None,
            )

    driver = DriverStub()
    graphiti = SimpleNamespace(driver=cast(GraphDriver, driver))

    states = await ZepGraphiti.find_episode_ingest_states(
        graphiti,  # type: ignore[arg-type]
        'publisher-graph',
        ['episode-1', 'episode-2'],
    )

    assert [state.uuid for state in states] == ['episode-1']
    assert states[0].group_id == 'another-graph'
    assert states[0].completed is True
    assert 'e.uuid IN $episode_uuids' in driver.query
    assert 'source_description CONTAINS' not in driver.query
    assert 'e.group_id = $group_id' not in driver.query
    assert driver.parameters == {'episode_uuids': ['episode-1', 'episode-2']}
