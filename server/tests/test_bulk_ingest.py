import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from graphiti_core.nodes import EpisodeType

import graph_service.routers.ingest as ingest_router
from graph_service.dto import AddMessagesBulkRequest, Message
from graph_service.routers.ingest import (
    _bulk_group_locks,
    _bulk_requests,
    add_messages_bulk,
    clear,
    delete_group,
)


def ingest_state(message: Message, *, completed: bool = True):
    return SimpleNamespace(
        uuid=message.uuid,
        name=message.name,
        content=f'{message.role or ""}({message.role_type}): {message.content}',
        source_description=message.source_description,
        source=EpisodeType.message.value,
        valid_at=message.timestamp,
        completed=completed,
    )


def test_only_change_set_and_graph_level_mutations_are_exposed() -> None:
    routes = {
        (route.path, frozenset(route.methods or set()))
        for route in ingest_router.router.routes
        if isinstance(route, APIRoute)
    }

    assert ('/messages/bulk', frozenset({'POST'})) in routes
    assert ('/group/{group_id}', frozenset({'DELETE'})) in routes
    assert ('/clear', frozenset({'POST'})) in routes
    legacy_paths = {'/messages', '/entity-node', '/entity-edge/{uuid}', '/episode/{uuid}'}
    assert all(path not in legacy_paths for path, _ in routes)


@pytest.fixture(autouse=True)
def clear_bulk_request_registry() -> None:
    _bulk_requests.clear()
    _bulk_group_locks.clear()


@pytest.mark.asyncio
async def test_add_messages_bulk_maps_messages_and_returns_completed_episodes() -> None:
    captured = {}

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return []

        async def add_episode_bulk(self, episodes, group_id):
            captured['episodes'] = episodes
            captured['group_id'] = group_id
            return SimpleNamespace(
                episodes=[SimpleNamespace(uuid='episode-1'), SimpleNamespace(uuid='episode-2')]
            )

        async def mark_episodes_completed(self, group_id, episode_uuids):
            captured['completed'] = (group_id, episode_uuids)

    request = AddMessagesBulkRequest(
        request_id='batch-1',
        group_id='publisher-graph',
        messages=[
            Message(
                uuid='episode-1',
                name='First',
                role_type='system',
                role='source',
                content='First article',
                source_description='first source',
            ),
            Message(
                uuid='episode-2',
                name='Second',
                role_type='user',
                role=None,
                content='Second article',
                source_description='second source',
            ),
        ],
    )

    response = await add_messages_bulk(request, GraphitiStub())  # type: ignore[arg-type]

    assert captured['group_id'] == 'publisher-graph'
    assert [episode.content for episode in captured['episodes']] == [
        'source(system): First article',
        '(user): Second article',
    ]
    assert all(episode.source == EpisodeType.message for episode in captured['episodes'])
    assert [episode.uuid for episode in captured['episodes']] == ['episode-1', 'episode-2']
    assert captured['completed'] == (
        'publisher-graph',
        ['episode-1', 'episode-2'],
    )
    assert response.success is True
    assert response.request_id == 'batch-1'
    assert response.episode_count == 2
    assert response.episode_uuids == ['episode-1', 'episode-2']
    assert response.processed_episode_uuids == ['episode-1', 'episode-2']


def test_add_messages_bulk_rejects_empty_batches() -> None:
    with pytest.raises(ValueError):
        AddMessagesBulkRequest(request_id='batch-1', group_id='publisher-graph', messages=[])


@pytest.mark.parametrize('group_id', ['', 'publisher graph', 'publisher/graph'])
def test_add_messages_bulk_rejects_invalid_graph_identifiers(group_id: str) -> None:
    with pytest.raises(ValueError):
        AddMessagesBulkRequest(
            request_id='batch-1',
            group_id=group_id,
            messages=[
                Message(
                    uuid='episode-1',
                    name='First',
                    role_type='system',
                    role='source',
                    content='First article',
                )
            ],
        )


def test_add_messages_bulk_requires_deterministic_episode_uuids() -> None:
    with pytest.raises(ValueError, match='must include deterministic episode UUIDs'):
        AddMessagesBulkRequest(
            request_id='batch-1',
            group_id='publisher-graph',
            messages=[
                Message(
                    name='First',
                    role_type='system',
                    role='source',
                    content='First article',
                )
            ],
        )


def test_add_messages_bulk_rejects_duplicate_episode_uuids() -> None:
    message = Message(
        uuid='episode-1',
        name='First',
        role_type='system',
        role='source',
        content='First article',
    )
    with pytest.raises(ValueError, match='must be unique'):
        AddMessagesBulkRequest(
            request_id='batch-1',
            group_id='publisher-graph',
            messages=[message, message],
        )


@pytest.mark.asyncio
async def test_mixed_batch_writes_only_missing_episodes_and_receipts_the_full_batch() -> None:
    captured = {}

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return [ingest_state(request.messages[0])]

        async def add_episode_bulk(self, episodes, group_id):
            captured['written'] = [episode.uuid for episode in episodes]
            return SimpleNamespace(episodes=[SimpleNamespace(uuid='episode-2')])

        async def mark_episodes_completed(self, group_id, episode_uuids):
            captured['marked'] = episode_uuids

    request = AddMessagesBulkRequest(
        request_id='batch-mixed',
        group_id='publisher-graph',
        messages=[
            Message(uuid='episode-1', name='First', role_type='system', role=None, content='First'),
            Message(
                uuid='episode-2', name='Second', role_type='system', role=None, content='Second'
            ),
        ],
    )

    response = await add_messages_bulk(request, GraphitiStub())  # type: ignore[arg-type]

    assert captured == {'written': ['episode-2'], 'marked': ['episode-2']}
    assert response.episode_count == 2
    assert response.episode_uuids == ['episode-1', 'episode-2']
    assert response.processed_episode_uuids == ['episode-2']


@pytest.mark.asyncio
async def test_concurrent_retries_join_the_same_in_flight_bulk_request() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return []

        async def add_episode_bulk(self, episodes, group_id):
            nonlocal call_count
            call_count += 1
            started.set()
            await release.wait()
            return SimpleNamespace(episodes=[SimpleNamespace(uuid='episode-1')])

        async def mark_episodes_completed(self, group_id, episode_uuids):
            pass

    request = AddMessagesBulkRequest(
        request_id='batch-1',
        group_id='publisher-graph',
        messages=[
            Message(
                uuid='episode-1',
                name='First',
                role_type='system',
                role='source',
                content='First',
            )
        ],
    )

    first = asyncio.create_task(add_messages_bulk(request, GraphitiStub()))  # type: ignore[arg-type]
    await started.wait()
    second = asyncio.create_task(add_messages_bulk(request, GraphitiStub()))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    release.set()

    first_result, second_result = await asyncio.gather(first, second)
    assert call_count == 1
    assert first_result == second_result


@pytest.mark.asyncio
async def test_bulk_request_id_cannot_be_reused_for_another_payload() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return []

        async def add_episode_bulk(self, episodes, group_id):
            started.set()
            await release.wait()
            return SimpleNamespace(episodes=[SimpleNamespace(uuid='episode-1')])

        async def mark_episodes_completed(self, group_id, episode_uuids):
            pass

    first = AddMessagesBulkRequest(
        request_id='batch-1',
        group_id='publisher-graph',
        messages=[
            Message(
                uuid='episode-1',
                name='First',
                role_type='system',
                role='source',
                content='First',
            )
        ],
    )
    changed = first.model_copy(
        update={
            'messages': [
                Message(
                    uuid='episode-1',
                    name='Changed',
                    role_type='system',
                    role='source',
                    content='Changed',
                )
            ]
        }
    )

    in_flight = asyncio.create_task(add_messages_bulk(first, GraphitiStub()))  # type: ignore[arg-type]
    await started.wait()
    with pytest.raises(HTTPException) as error:
        await add_messages_bulk(changed, GraphitiStub())  # type: ignore[arg-type]

    assert error.value.status_code == 409
    release.set()
    await in_flight


@pytest.mark.asyncio
async def test_completed_request_state_is_evicted_so_a_reset_can_be_retried() -> None:
    call_count = 0

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return []

        async def add_episode_bulk(self, episodes, group_id):
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(episodes=[SimpleNamespace(uuid='episode-1')])

        async def mark_episodes_completed(self, group_id, episode_uuids):
            pass

    request = AddMessagesBulkRequest(
        request_id='batch-1',
        group_id='publisher-graph',
        messages=[
            Message(
                uuid='episode-1',
                name='First',
                role_type='system',
                role='source',
                content='First',
            )
        ],
    )

    await add_messages_bulk(request, GraphitiStub())  # type: ignore[arg-type]
    await asyncio.sleep(0)
    assert _bulk_requests == {}

    await add_messages_bulk(request, GraphitiStub())  # type: ignore[arg-type]
    assert call_count == 2


@pytest.mark.asyncio
async def test_durable_episode_identity_rejects_changed_semantic_payload() -> None:
    original = Message(
        uuid='episode-1',
        name='First',
        role_type='system',
        role='source',
        content='Original',
    )

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return [ingest_state(original)]

        async def add_episode_bulk(self, episodes, group_id):
            raise AssertionError('conflicting payload must not be processed')

    request = AddMessagesBulkRequest(
        request_id='changed-batch',
        group_id='publisher-graph',
        messages=[original.model_copy(update={'content': 'Changed'})],
    )

    with pytest.raises(HTTPException) as error:
        await add_messages_bulk(request, GraphitiStub())  # type: ignore[arg-type]

    assert error.value.status_code == 409
    assert error.value.detail['episode_uuids'] == ['episode-1']


@pytest.mark.asyncio
async def test_matching_incomplete_episode_is_reprocessed() -> None:
    message = Message(
        uuid='episode-1',
        name='First',
        role_type='system',
        role='source',
        content='Original',
    )
    processed = False

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return [ingest_state(message, completed=False)]

        async def add_episode_bulk(self, episodes, group_id):
            nonlocal processed
            processed = True
            return SimpleNamespace(episodes=[SimpleNamespace(uuid='episode-1')])

        async def mark_episodes_completed(self, group_id, episode_uuids):
            pass

    request = AddMessagesBulkRequest(
        request_id='retry-batch', group_id='publisher-graph', messages=[message]
    )
    response = await add_messages_bulk(request, GraphitiStub())  # type: ignore[arg-type]

    assert processed is True
    assert response.processed_episode_uuids == ['episode-1']


@pytest.mark.asyncio
async def test_failed_semantic_write_is_not_marked_complete_and_can_be_retried() -> None:
    attempts = 0
    completion_calls = 0

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return []

        async def add_episode_bulk(self, episodes, group_id):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError('final semantic write failed')
            return SimpleNamespace(episodes=[SimpleNamespace(uuid='episode-1')])

        async def mark_episodes_completed(self, group_id, episode_uuids):
            nonlocal completion_calls
            completion_calls += 1

    request = AddMessagesBulkRequest(
        request_id='batch-1',
        group_id='publisher-graph',
        messages=[
            Message(
                uuid='episode-1',
                name='First',
                role_type='system',
                role='source',
                content='First',
            )
        ],
    )

    with pytest.raises(RuntimeError, match='final semantic write failed'):
        await add_messages_bulk(request, GraphitiStub())  # type: ignore[arg-type]
    await asyncio.sleep(0)
    assert completion_calls == 0
    assert _bulk_requests == {}

    result = await add_messages_bulk(request, GraphitiStub())  # type: ignore[arg-type]
    assert result.success is True
    assert attempts == 2
    assert completion_calls == 1


@pytest.mark.asyncio
async def test_sequential_replay_uses_durable_completion_without_semantic_work() -> None:
    completed: set[str] = set()
    add_count = 0

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return [
                ingest_state(request.messages[0]) for uuid in episode_uuids if uuid in completed
            ]

        async def add_episode_bulk(self, episodes, group_id):
            nonlocal add_count
            add_count += 1
            return SimpleNamespace(
                episodes=[SimpleNamespace(uuid=episode.uuid) for episode in episodes]
            )

        async def mark_episodes_completed(self, group_id, episode_uuids):
            completed.update(episode_uuids)

    request = AddMessagesBulkRequest(
        request_id='batch-1',
        group_id='publisher-graph',
        messages=[
            Message(
                uuid='episode-1',
                name='First',
                role_type='system',
                role='source',
                content='First',
            )
        ],
    )

    await add_messages_bulk(request, GraphitiStub())  # type: ignore[arg-type]
    await asyncio.sleep(0)
    replay = request.model_copy(update={'request_id': 'different-request-partition'})
    result = await add_messages_bulk(replay, GraphitiStub())  # type: ignore[arg-type]

    assert add_count == 1
    assert result.episode_uuids == ['episode-1']
    assert result.processed_episode_uuids == []


@pytest.mark.asyncio
async def test_overlapping_batches_serialize_by_graph_and_skip_completed_episodes() -> None:
    completed: set[str] = set()
    first_write_started = asyncio.Event()
    release_first_write = asyncio.Event()
    add_count = 0

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return [ingest_state(first.messages[0]) for uuid in episode_uuids if uuid in completed]

        async def add_episode_bulk(self, episodes, group_id):
            nonlocal add_count
            add_count += 1
            first_write_started.set()
            await release_first_write.wait()
            return SimpleNamespace(
                episodes=[SimpleNamespace(uuid=episode.uuid) for episode in episodes]
            )

        async def mark_episodes_completed(self, group_id, episode_uuids):
            completed.update(episode_uuids)

    first = AddMessagesBulkRequest(
        request_id='batch-1',
        group_id='publisher-graph',
        messages=[
            Message(
                uuid='episode-1',
                name='First',
                role_type='system',
                role='source',
                content='First',
            )
        ],
    )
    overlapping = first.model_copy(update={'request_id': 'different-request-partition'})

    first_task = asyncio.create_task(add_messages_bulk(first, GraphitiStub()))  # type: ignore[arg-type]
    await first_write_started.wait()
    overlap_task = asyncio.create_task(add_messages_bulk(overlapping, GraphitiStub()))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    assert add_count == 1
    release_first_write.set()

    first_result, overlap_result = await asyncio.gather(first_task, overlap_task)
    assert add_count == 1
    assert first_result.episode_uuids == overlap_result.episode_uuids == ['episode-1']


@pytest.mark.asyncio
async def test_group_delete_waits_for_in_flight_bulk_ingestion() -> None:
    write_started = asyncio.Event()
    release_write = asyncio.Event()
    delete_called = asyncio.Event()

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return []

        async def add_episode_bulk(self, episodes, group_id):
            write_started.set()
            await release_write.wait()
            return SimpleNamespace(
                episodes=[SimpleNamespace(uuid=episode.uuid) for episode in episodes]
            )

        async def mark_episodes_completed(self, group_id, episode_uuids):
            pass

        async def delete_group(self, group_id):
            delete_called.set()

    graphiti = GraphitiStub()
    request = AddMessagesBulkRequest(
        request_id='batch-1',
        group_id='publisher-graph',
        messages=[
            Message(
                uuid='episode-1',
                name='First',
                role_type='system',
                role='source',
                content='First',
            )
        ],
    )

    write_task = asyncio.create_task(add_messages_bulk(request, graphiti))  # type: ignore[arg-type]
    await write_started.wait()
    delete_task = asyncio.create_task(delete_group('publisher-graph', graphiti))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    assert not delete_called.is_set()

    release_write.set()
    await asyncio.gather(write_task, delete_task)
    assert delete_called.is_set()


@pytest.mark.asyncio
async def test_queued_graph_does_not_block_another_graph() -> None:
    first_graph_started = asyncio.Event()
    second_graph_started = asyncio.Event()
    release_first_graph = asyncio.Event()

    class GraphitiStub:
        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return []

        async def add_episode_bulk(self, episodes, group_id):
            if group_id == 'graph-a':
                first_graph_started.set()
                await release_first_graph.wait()
            else:
                second_graph_started.set()
            return SimpleNamespace(episodes=[SimpleNamespace(uuid=episodes[0].uuid)])

        async def mark_episodes_completed(self, group_id, episode_uuids):
            pass

    def request(request_id: str, group_id: str, episode_uuid: str):
        return AddMessagesBulkRequest(
            request_id=request_id,
            group_id=group_id,
            messages=[
                Message(
                    uuid=episode_uuid,
                    name=episode_uuid,
                    role_type='system',
                    role='source',
                    content=episode_uuid,
                )
            ],
        )

    first = asyncio.create_task(
        add_messages_bulk(request('a-1', 'graph-a', 'episode-a-1'), GraphitiStub())  # type: ignore[arg-type]
    )
    await first_graph_started.wait()
    queued = asyncio.create_task(
        add_messages_bulk(request('a-2', 'graph-a', 'episode-a-2'), GraphitiStub())  # type: ignore[arg-type]
    )
    await asyncio.sleep(0)
    independent = asyncio.create_task(
        add_messages_bulk(request('b-1', 'graph-b', 'episode-b-1'), GraphitiStub())  # type: ignore[arg-type]
    )

    await asyncio.wait_for(second_graph_started.wait(), timeout=2)
    release_first_graph.set()
    await asyncio.gather(first, queued, independent)


@pytest.mark.asyncio
async def test_global_clear_waits_for_in_flight_bulk_ingestion(monkeypatch) -> None:
    write_started = asyncio.Event()
    release_write = asyncio.Event()
    clear_called = asyncio.Event()

    class GraphitiStub:
        driver = SimpleNamespace()

        async def find_episode_ingest_states(self, group_id, episode_uuids):
            return []

        async def add_episode_bulk(self, episodes, group_id):
            write_started.set()
            await release_write.wait()
            return SimpleNamespace(
                episodes=[SimpleNamespace(uuid=episode.uuid) for episode in episodes]
            )

        async def mark_episodes_completed(self, group_id, episode_uuids):
            pass

        async def build_indices_and_constraints(self):
            pass

    async def clear_data_stub(driver):
        clear_called.set()

    monkeypatch.setattr(ingest_router, 'clear_data', clear_data_stub)
    graphiti = GraphitiStub()
    request = AddMessagesBulkRequest(
        request_id='batch-1',
        group_id='publisher-graph',
        messages=[
            Message(
                uuid='episode-1',
                name='First',
                role_type='system',
                role='source',
                content='First',
            )
        ],
    )

    write_task = asyncio.create_task(add_messages_bulk(request, graphiti))  # type: ignore[arg-type]
    await write_started.wait()
    clear_task = asyncio.create_task(clear(graphiti))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    assert not clear_called.is_set()

    release_write.set()
    await asyncio.gather(write_task, clear_task)
    assert clear_called.is_set()
