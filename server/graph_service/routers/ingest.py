import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from weakref import WeakValueDictionary

from fastapi import APIRouter, HTTPException, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.bulk_utils import RawEpisode  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import (
    AddMessagesBulkRequest,
    AddMessagesBulkResponse,
    Message,
    Result,
)
from graph_service.zep_graphiti import EpisodeIngestState, ZepGraphitiDep

router = APIRouter()


@dataclass(frozen=True)
class BulkRequestState:
    payload_hash: str
    task: asyncio.Task[AddMessagesBulkResponse]


_bulk_request_lock = asyncio.Lock()
_bulk_requests: dict[str, BulkRequestState] = {}
_bulk_group_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
_bulk_episode_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
_bulk_clear_barrier = asyncio.Lock()


def _evict_bulk_request(request_id: str, task: asyncio.Task[AddMessagesBulkResponse]) -> None:
    state = _bulk_requests.get(request_id)
    if state is not None and state.task is task:
        _bulk_requests.pop(request_id, None)


def _bulk_group_lock(group_id: str) -> asyncio.Lock:
    lock = _bulk_group_locks.get(group_id)
    if lock is None:
        lock = asyncio.Lock()
        _bulk_group_locks[group_id] = lock
    return lock


def _bulk_episode_lock(episode_uuid: str) -> asyncio.Lock:
    lock = _bulk_episode_locks.get(episode_uuid)
    if lock is None:
        lock = asyncio.Lock()
        _bulk_episode_locks[episode_uuid] = lock
    return lock


@asynccontextmanager
async def _locked_bulk_group(group_id: str, episode_uuids: list[str]):
    async with _bulk_clear_barrier:
        group_lock = _bulk_group_lock(group_id)
        episode_locks = [_bulk_episode_lock(uuid) for uuid in sorted(set(episode_uuids))]
    acquired: list[asyncio.Lock] = []
    try:
        await group_lock.acquire()
        acquired.append(group_lock)
        for lock in episode_locks:
            await lock.acquire()
            acquired.append(lock)
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()


def _episode_body(message: Message) -> str:
    return f'{message.role or ""}({message.role_type}): {message.content}'


def _episode_matches_message(
    state: EpisodeIngestState, message: Message, group_id: str
) -> bool:
    return (
        state.group_id == group_id
        and state.name == message.name
        and state.content == _episode_body(message)
        and state.source_description == message.source_description
        and state.source == EpisodeType.message.value
        and state.valid_at == message.timestamp
    )


@router.post(
    '/messages/bulk',
    status_code=status.HTTP_201_CREATED,
    response_model=AddMessagesBulkResponse,
)
async def add_messages_bulk(
    request: AddMessagesBulkRequest,
    graphiti: ZepGraphitiDep,
) -> AddMessagesBulkResponse:
    payload_hash = hashlib.sha256(
        json.dumps(request.model_dump(mode='json'), sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()
    async with _bulk_request_lock:
        existing = _bulk_requests.get(request.request_id)
        if existing is not None and existing.payload_hash != payload_hash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail='request_id was already used for a different bulk payload',
            )
        if existing is None:
            task = asyncio.create_task(_apply_messages_bulk(request, graphiti))
            existing = BulkRequestState(payload_hash=payload_hash, task=task)
            _bulk_requests[request.request_id] = existing
            task.add_done_callback(partial(_evict_bulk_request, request.request_id))
    try:
        return await asyncio.shield(existing.task)
    except asyncio.CancelledError:
        # Keep the request-scoped Graphiti client alive until the shared task
        # finishes; a retry with the same request_id will join this task.
        await existing.task
        raise


async def _apply_messages_bulk(
    request: AddMessagesBulkRequest,
    graphiti: ZepGraphitiDep,
) -> AddMessagesBulkResponse:
    requested_uuids = [message.uuid for message in request.messages if message.uuid is not None]
    async with _locked_bulk_group(request.group_id, requested_uuids):
        states = {
            state.uuid: state
            for state in await graphiti.find_episode_ingest_states(
                request.group_id, requested_uuids
            )
        }
        conflicting_uuids: list[str] = []
        for message in request.messages:
            assert message.uuid is not None
            state = states.get(message.uuid)
            if state is not None and not _episode_matches_message(
                state, message, request.group_id
            ):
                conflicting_uuids.append(message.uuid)
        if conflicting_uuids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    'message': 'episode UUID was already used for a different semantic payload',
                    'episode_uuids': conflicting_uuids,
                },
            )
        completed_uuids = {uuid for uuid, state in states.items() if state.completed}
        missing_messages = [
            message for message in request.messages if message.uuid not in completed_uuids
        ]
        if missing_messages:
            result = await graphiti.add_episode_bulk(
                [
                    RawEpisode(
                        uuid=message.uuid,
                        name=message.name,
                        content=_episode_body(message),
                        reference_time=message.timestamp,
                        source=EpisodeType.message,
                        source_description=message.source_description,
                    )
                    for message in missing_messages
                ],
                group_id=request.group_id,
            )
            written_uuids = [episode.uuid for episode in result.episodes]
            expected_uuids = [
                message.uuid for message in missing_messages if message.uuid is not None
            ]
            if sorted(written_uuids) != sorted(expected_uuids):
                raise RuntimeError('Graphiti bulk write returned unexpected episode identities')
            await graphiti.mark_episodes_completed(request.group_id, written_uuids)
    return AddMessagesBulkResponse(
        success=True,
        request_id=request.request_id,
        message='Messages added in bulk',
        episode_count=len(requested_uuids),
        episode_uuids=requested_uuids,
        processed_episode_uuids=[
            message.uuid for message in missing_messages if message.uuid is not None
        ],
    )


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    async with _locked_bulk_group(group_id, []):
        await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
):
    async with _bulk_clear_barrier:
        locks = list(_bulk_group_locks.values())
        for lock in locks:
            await lock.acquire()
        try:
            await clear_data(graphiti.driver)
            await graphiti.build_indices_and_constraints()
        finally:
            for lock in reversed(locks):
                lock.release()
    return Result(message='Graph cleared', success=True)
