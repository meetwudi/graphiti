import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.driver.driver import GraphDriver, GraphProvider  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.helpers import parse_db_date  # type: ignore
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


def _driver_for_group(driver: GraphDriver, group_id: str) -> GraphDriver:
    return driver.with_database(group_id) if driver.provider == GraphProvider.FALKORDB else driver


@dataclass(frozen=True)
class EpisodeIngestState:
    uuid: str
    name: str
    content: str
    source_description: str
    source: str
    valid_at: datetime
    completed: bool


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: LLMClient | None = None,
        **kwargs,
    ):
        super().__init__(uri, user, password, llm_client, **kwargs)  # type: ignore

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def find_episode_ingest_states(
        self, group_id: str, episode_uuids: list[str]
    ) -> list[EpisodeIngestState]:
        driver = _driver_for_group(self.driver, group_id)
        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic)
            WHERE e.group_id = $group_id
              AND e.uuid IN $episode_uuids
            RETURN e.uuid AS uuid,
                   e.name AS name,
                   e.content AS content,
                   e.source_description AS source_description,
                   e.source AS source,
                   e.valid_at AS valid_at,
                   coalesce(e.flint_ingest_completed, false) AS completed
            """,
            group_id=group_id,
            episode_uuids=episode_uuids,
        )
        states = []
        for record in records:
            valid_at = parse_db_date(record['valid_at'])
            if valid_at is None:
                raise RuntimeError(f'episode {record["uuid"]} has no valid_at timestamp')
            states.append(EpisodeIngestState(**{**record, 'valid_at': valid_at}))
        return states

    async def mark_episodes_completed(self, group_id: str, episode_uuids: list[str]) -> None:
        driver = _driver_for_group(self.driver, group_id)
        records, _, _ = await driver.execute_query(
            """
            MATCH (e:Episodic)
            WHERE e.group_id = $group_id AND e.uuid IN $episode_uuids
            SET e.flint_ingest_completed = true
            RETURN count(e) AS completed_count
            """,
            group_id=group_id,
            episode_uuids=episode_uuids,
        )
        completed_count = records[0]['completed_count'] if records else 0
        if completed_count != len(episode_uuids):
            raise RuntimeError(f'completed {completed_count} of {len(episode_uuids)} bulk episodes')

    async def delete_group(self, group_id: str):
        driver = _driver_for_group(self.driver, group_id)

        try:
            edges = await EntityEdge.get_by_group_ids(driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(driver, [group_id])

        for edge in edges:
            await edge.delete(driver)

        for node in nodes:
            await node.delete(driver)

        for episode in episodes:
            await episode.delete(driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


def _create_graphiti_client(settings: ZepEnvDep) -> ZepGraphiti:
    """Create a ZepGraphiti client based on the configured database backend."""
    if settings.db_backend == 'falkordb':
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(  # type: ignore
            host=settings.falkordb_host or 'localhost',  # type: ignore
            port=settings.falkordb_port or 6379,  # type: ignore
            database=settings.falkordb_database or 'default_db',  # type: ignore
        )
        return ZepGraphiti(graph_driver=driver)  # type: ignore
    else:
        # Validate Neo4j settings are present
        if not all([settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password]):
            raise ValueError(
                'Neo4j configuration (neo4j_uri, neo4j_user, neo4j_password) is required '
                "when db_backend is 'neo4j'"
            )
        return ZepGraphiti(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )


async def get_graphiti(settings: ZepEnvDep):
    client = _create_graphiti_client(settings)
    if settings.openai_base_url is not None:
        client.llm_client.config.base_url = settings.openai_base_url
    if settings.openai_api_key is not None:
        client.llm_client.config.api_key = settings.openai_api_key
    if settings.model_name is not None:
        client.llm_client.model = settings.model_name

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = _create_graphiti_client(settings)
    try:
        await client.build_indices_and_constraints()
    finally:
        await client.close()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        episodes=edge.episodes or [],
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
