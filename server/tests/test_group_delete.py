from typing import cast

from graphiti_core.driver.driver import GraphDriver, GraphProvider

from graph_service.zep_graphiti import _driver_for_group


class DriverStub:
    def __init__(self, provider: GraphProvider):
        self.provider = provider
        self.selected_groups: list[str] = []

    def with_database(self, group_id: str):
        self.selected_groups.append(group_id)
        return object()


def test_falkordb_delete_selects_the_group_database_without_mutating_the_client() -> None:
    driver = DriverStub(GraphProvider.FALKORDB)

    selected = _driver_for_group(cast(GraphDriver, driver), 'publisher-graph')

    assert selected is not driver
    assert driver.selected_groups == ['publisher-graph']


def test_neo4j_delete_keeps_the_configured_database() -> None:
    driver = DriverStub(GraphProvider.NEO4J)

    selected = _driver_for_group(cast(GraphDriver, driver), 'publisher-group')

    assert selected is driver
    assert driver.selected_groups == []
