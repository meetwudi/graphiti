from pathlib import Path

import tomllib

SERVER_ROOT = Path(__file__).resolve().parents[1]
GRAPHITI_ROOT = SERVER_ROOT.parent


def test_server_lock_resolves_graphiti_core_from_vendor_checkout() -> None:
    project = tomllib.loads((SERVER_ROOT / 'pyproject.toml').read_text())
    lock = (SERVER_ROOT / 'uv.lock').read_text()

    assert project['tool']['uv']['sources']['graphiti-core'] == {'path': '..'}
    assert project['project']['optional-dependencies']['falkordb'] == [
        'graphiti-core[falkordb]>=0.28.2'
    ]
    assert 'name = "graphiti-core"\nversion = "0.29.3"\nsource = { directory = "../" }' in lock


def test_docker_image_copies_local_core_and_does_not_install_it_from_pypi() -> None:
    dockerfile = (GRAPHITI_ROOT / 'Dockerfile').read_text()

    assert 'COPY ./graphiti_core ./graphiti_core' in dockerfile
    assert 'COPY ./server/pyproject.toml' in dockerfile
    assert 'pip install --upgrade graphiti-core' not in dockerfile
    assert 'graphiti-core==' not in dockerfile
    assert 'sync --frozen --no-dev --extra falkordb' in dockerfile
    assert 'pip install --upgrade "/app[falkordb]"' not in dockerfile
