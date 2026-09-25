"""Common backend contract tests.

These tests verify the same behavior across every locally available backend (memory, SQLite, and
filesystem), without requiring any external database services.  Behavior is expressed in terms of
logical records (cache keys mapping to responses, plus redirect aliases mapping to those keys)
rather than backend-internal storage layouts.

Expiration is always driven by a fake clock (``time-machine``); no test in this module sleeps or
waits for TTLs based on wall-clock time.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sqlite3
import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest

from requests_cache.backends import BaseCache, FileCache, SQLiteCache
from requests_cache.models import CachedRequest, CachedResponse
from requests_cache.policy import utcnow
from requests_cache.serializers import pickle_serializer

try:
    from time_machine import travel as time_travel
except ImportError:  # pragma: no cover
    pytest.skip('time-machine is required for backend contract tests', allow_module_level=True)


EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
FIXED_SEED = 1318
N_OPERATIONS = 60


@dataclass
class BackendSpec:
    """Definition of a backend participating in the shared contract tests."""

    name: str
    backend_factory: Callable[[str], BaseCache]
    persistent: bool
    corruptable: bool
    serializer: object = None


def _make_memory_cache(tmp_path: str) -> BaseCache:
    return BaseCache(str(tmp_path))


def _make_sqlite_cache(tmp_path: str) -> BaseCache:
    return SQLiteCache(os.path.join(tmp_path, 'cache.sqlite'))


def _make_filesystem_cache(tmp_path: str) -> BaseCache:
    # Use the pickle serializer so response bodies round-trip byte-for-byte
    cache = FileCache(os.path.join(tmp_path, 'files'))
    cache.responses.serializer = pickle_serializer
    cache.responses.is_binary = True
    return cache


BACKEND_SPECS = [
    BackendSpec('memory', _make_memory_cache, persistent=False, corruptable=False),
    BackendSpec('sqlite', _make_sqlite_cache, persistent=True, corruptable=True),
    BackendSpec('filesystem', _make_filesystem_cache, persistent=True, corruptable=True),
]


# --------------------------------------------------------------------------------------
# Helpers and reference model
# --------------------------------------------------------------------------------------


def make_response(
    generation: int,
    *,
    url: Optional[str] = None,
    expires: Optional[datetime] = None,
    status_code: int = 200,
) -> CachedResponse:
    """Create a self-contained cached response whose body encodes its generation."""
    url = url or f'https://contract.test/item/{generation}'
    content = json.dumps({'generation': generation, 'payload': str(generation) * 4}).encode()
    return CachedResponse(
        url=url,
        status_code=status_code,
        headers={'Content-Type': 'application/json'},
        expires=expires,
        request=CachedRequest(method='GET', url=url),
        content=content,
    )


def make_redirect_response(
    target_key: str,
    target_url: str,
    *,
    expires: Optional[datetime] = None,
    status_code: int = 302,
) -> CachedResponse:
    """Create a minimal redirect response whose cache key can be derived deterministically."""
    url = f'https://contract.test/redirect/{target_key}'
    return CachedResponse(
        url=url,
        status_code=status_code,
        headers={'Location': target_url},
        expires=expires,
        request=CachedRequest(method='GET', url=url),
        content=b'',
    )


def redirect_key(cache: BaseCache, response: CachedResponse) -> str:
    return cache.create_key(response.request)


@dataclass
class LogicalRecord:
    """The logical state of one cached response."""

    response: CachedResponse


@dataclass
class ReferenceModel:
    """Reference implementation of the logical record contract shared by all backends."""

    records: Dict[str, CachedResponse] = field(default_factory=dict)
    redirects: Dict[str, str] = field(default_factory=dict)

    def save(self, key: str, response: CachedResponse):
        response.cache_key = key
        self.records[key] = response

    def add_redirect(self, alias: str, target: str):
        self.redirects[alias] = target
        self._prune_redirects()

    def delete(self, *keys: str):
        for key in keys:
            self.records.pop(key, None)
        self._prune_redirects()

    def clear(self):
        self.records.clear()
        self.redirects.clear()

    def remove_expired(self, now: datetime):
        expired = [
            key
            for key, response in self.records.items()
            if response.expires is not None and response.expires <= now
        ]
        for key in expired:
            del self.records[key]
        self._prune_redirects()

    def remove_invalid(self, valid_keys):
        """Drop redirects/records that the backend marks invalid (deserialization failures)."""
        self.records = {key: value for key, value in self.records.items() if key in valid_keys}
        self._prune_redirects()

    def _prune_redirects(self):
        for alias in [a for a, target in self.redirects.items() if target not in self.records]:
            del self.redirects[alias]


def assert_same_response(actual: CachedResponse, expected: CachedResponse):
    assert actual is not None
    assert actual.status_code == expected.status_code
    assert actual.url == expected.url
    assert actual.content == expected.content
    # Compare at whole-second resolution, mirroring the HTTP/unix timestamp semantics used by
    # several backends.
    assert int(actual.created_at.timestamp()) == int(expected.created_at.timestamp())
    if expected.expires is None:
        assert actual.expires is None
    else:
        assert int(actual.expires.timestamp()) == int(expected.expires.timestamp())


def assert_contract(cache: BaseCache, reference: ReferenceModel):
    """Compare a backend's logical records against the reference model."""
    response_keys = set(cache.responses.keys())
    assert response_keys == set(reference.records.keys())
    assert len(cache.responses) == len(reference.records)

    for key, expected in reference.records.items():
        assert key in cache.responses
        assert cache.contains(key=key)
        actual = cache.get_response(key)
        assert_same_response(actual, expected)
        assert actual.cache_key == key

    # A redirect alias points at exactly one existing response body
    assert dict(cache.redirects) == reference.redirects
    for alias, target in reference.redirects.items():
        assert target in cache.responses, f'redirect {alias!r} points to missing body {target!r}'
        assert alias not in cache.responses
        followed = cache.get_response(alias)
        assert followed is not None
        assert followed.cache_key == target
        assert cache.contains(key=alias)

    # No orphan redirects: the backend must not expose redirects to missing bodies
    for alias in cache.redirects.keys():
        assert cache.redirects[alias] in cache.responses


def fresh_cache(spec: BackendSpec, tmp_path: Path) -> BaseCache:
    path = str(tmp_path / spec.name)
    cache = spec.backend_factory(path)
    cache.clear()
    return cache


def reopen_cache(spec: BackendSpec, tmp_path: Path) -> BaseCache:
    """Reopen the same cache location; for non-persistent backends this is a new empty cache."""
    path = str(tmp_path / spec.name)
    cache = spec.backend_factory(path)
    if not spec.persistent:
        cache.clear()
    return cache


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(params=BACKEND_SPECS, ids=[spec.name for spec in BACKEND_SPECS])
def spec(request) -> BackendSpec:
    return request.param


@pytest.fixture
def cache(spec, tmp_path):
    instance = fresh_cache(spec, tmp_path)
    yield instance
    instance.close()


# --------------------------------------------------------------------------------------
# Basic save/get/contains/delete/clear contract
# --------------------------------------------------------------------------------------


class TestBasicContract:
    def test_save_get_contains(self, cache):
        response = make_response(1)
        cache.responses['key_1'] = response

        assert 'key_1' in cache.responses
        assert cache.contains(key='key_1')
        assert_same_response(cache.get_response('key_1'), response)
        assert cache.get_response('missing', default=None) is None

    def test_overwrite_last_write_wins(self, cache):
        first = make_response(1)
        second = make_response(2)
        cache.responses['key_1'] = first
        cache.responses['key_1'] = second

        assert len(cache.responses) == 1
        assert_same_response(cache.get_response('key_1'), second)

    def test_delete(self, cache):
        for i in range(5):
            cache.responses[f'key_{i}'] = make_response(i)
        cache.delete('key_1', 'key_3', 'missing_key')

        assert set(cache.responses.keys()) == {'key_0', 'key_2', 'key_4'}
        with pytest.raises(KeyError):
            del cache.responses['key_1']
        with pytest.raises(KeyError):
            cache.responses['key_1']

    def test_bulk_delete(self, cache):
        for i in range(20):
            cache.responses[f'key_{i}'] = make_response(i)
        cache.responses.bulk_delete([f'key_{i}' for i in range(10)])
        cache.responses.bulk_delete(['nonexistent_key'])

        remaining = {f'key_{i}' for i in range(10, 20)}
        assert set(cache.responses.keys()) == remaining
        assert len(cache.responses) == 10

    def test_clear(self, cache):
        for i in range(5):
            cache.responses[f'key_{i}'] = make_response(i)
        cache.redirects['alias'] = 'key_0'
        cache.clear()

        assert len(cache.responses) == 0
        assert len(cache.redirects) == 0
        assert 'key_0' not in cache.responses


# --------------------------------------------------------------------------------------
# Expiration, driven entirely by a fake clock
# --------------------------------------------------------------------------------------


class TestExpirationContract:
    def test_expiration_with_fake_clock(self, cache):
        reference = ReferenceModel()

        with time_travel(EPOCH) as traveller:
            cache.responses['fresh'] = make_response(1, url='https://contract.test/fresh',
                                                    expires=EPOCH + timedelta(seconds=10))
            reference.save('fresh', cache.responses['fresh'])
            cache.responses['expired'] = make_response(2, url='https://contract.test/expired',
                                                       expires=EPOCH - timedelta(seconds=1))
            reference.save('expired', cache.responses['expired'])
            cache.responses['eternal'] = make_response(3, url='https://contract.test/eternal')
            reference.save('eternal', cache.responses['eternal'])

            assert_contract(cache, reference)
            assert not cache.get_response('fresh').is_expired
            assert cache.get_response('expired').is_expired
            assert not cache.get_response('eternal').is_expired

            traveller.shift(20)
            reference.remove_expired(utcnow())
            cache.delete(expired=True)
            assert_contract(cache, reference)
            assert set(cache.responses.keys()) == {'eternal'}

    def test_remove_expired_in_stages(self, cache):
        reference = ReferenceModel()
        with time_travel(EPOCH) as traveller:
            for i, ttl in enumerate([5, 15, 30]):
                key = f'key_{i}'
                response = make_response(i, expires=EPOCH + timedelta(seconds=ttl))
                cache.responses[key] = response
                reference.save(key, response)

            traveller.shift(10)
            cache.delete(expired=True)
            reference.remove_expired(utcnow())
            assert_contract(cache, reference)
            assert set(cache.responses.keys()) == {'key_1', 'key_2'}

            traveller.shift(10)
            cache.delete(expired=True)
            reference.remove_expired(utcnow())
            assert_contract(cache, reference)
            assert set(cache.responses.keys()) == {'key_2'}

            traveller.shift(20)
            cache.delete(expired=True)
            reference.remove_expired(utcnow())
            assert_contract(cache, reference)
            assert set(cache.responses.keys()) == set()

    def test_remove_expired_responses_deprecated_alias(self, cache):
        """The deprecated remove_expired_responses() alias must honor the same contract."""
        with time_travel(EPOCH):
            cache.responses['expired'] = make_response(1, expires=EPOCH - timedelta(seconds=1))
            cache.responses['fresh'] = make_response(2, expires=EPOCH + timedelta(seconds=30))
        with time_travel(EPOCH):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                cache.remove_expired_responses()
        assert set(cache.responses.keys()) == {'fresh'}


# --------------------------------------------------------------------------------------
# Redirect keys
# --------------------------------------------------------------------------------------


class TestRedirectContract:
    def test_redirect_alias_resolves_to_body(self, cache):
        reference = ReferenceModel()
        with time_travel(EPOCH):
            target = make_response(1, url='https://contract.test/final')
            cache.responses['target_key'] = target
            reference.save('target_key', target)

            redirect = make_redirect_response('target_key', target.url)
            alias = redirect_key(cache, redirect)
            cache.redirects[alias] = 'target_key'
            reference.add_redirect(alias, 'target_key')

            assert_contract(cache, reference)

    def test_deleting_body_prunes_redirect(self, cache):
        reference = ReferenceModel()
        with time_travel(EPOCH):
            target = make_response(1)
            cache.responses['target_key'] = target
            reference.save('target_key', target)
            redirect = make_redirect_response('target_key', target.url)
            alias = redirect_key(cache, redirect)
            cache.redirects[alias] = 'target_key'
            reference.add_redirect(alias, 'target_key')

            cache.delete('target_key')
            reference.delete('target_key')
            assert_contract(cache, reference)
            assert alias not in cache.redirects
            assert cache.get_response(alias, default=None) is None


# --------------------------------------------------------------------------------------
# Close / reopen persistence
# --------------------------------------------------------------------------------------


class TestReopenContract:
    def test_records_survive_reopen(self, spec, tmp_path):
        cache = fresh_cache(spec, tmp_path)
        reference = ReferenceModel()
        with time_travel(EPOCH):
            for i in range(5):
                key = f'key_{i}'
                response = make_response(i, expires=EPOCH + timedelta(hours=1))
                cache.responses[key] = response
                reference.save(key, response)
            cache.responses['eternal'] = make_response(99, url='https://contract.test/eternal')
            reference.save('eternal', cache.responses['eternal'])
            alias_target = make_response(7, url='https://contract.test/redirected')
            cache.responses['alias_target'] = alias_target
            reference.save('alias_target', alias_target)
            redirect = make_redirect_response('alias_target', alias_target.url)
            alias = redirect_key(cache, redirect)
            cache.redirects[alias] = 'alias_target'
            reference.add_redirect(alias, 'alias_target')

            assert_contract(cache, reference)
            cache.close()

        reopened = reopen_cache(spec, tmp_path)
        try:
            with time_travel(EPOCH + timedelta(minutes=30)):
                if spec.persistent:
                    assert_contract(reopened, reference)
                else:
                    assert len(reopened.responses) == 0
                    assert len(reopened.redirects) == 0

                if spec.persistent:
                    with time_travel(EPOCH + timedelta(hours=2)):
                        reopened.delete(expired=True)
                    # Records created without an expiry, plus the redirect chain, never expire
                    assert set(reopened.responses.keys()) == {'eternal', 'alias_target'}
                    assert alias in reopened.redirects
        finally:
            reopened.close()


# --------------------------------------------------------------------------------------
# Deterministic operation-sequence model test
# --------------------------------------------------------------------------------------


class TestOperationSequenceContract:
    def test_seeded_operation_sequence(self, spec, tmp_path):
        """Run a fixed-seed sequence of operations against a reference model, checking after each
        step, then verify again after close/reopen."""
        cache = fresh_cache(spec, tmp_path)
        reference = ReferenceModel()
        rng = random.Random(FIXED_SEED)
        known_keys: List[str] = []

        def refresh_known_keys():
            known_keys[:] = sorted(reference.records.keys())

        with time_travel(EPOCH) as traveller:
            for step in range(N_OPERATIONS):
                action = rng.choice(
                    ['save', 'save', 'save', 'delete', 'bulk_delete', 'overwrite',
                     'expired_save', 'advance', 'noop']
                )

                if action == 'save' or action == 'expired_save':
                    key = f'key_{rng.randrange(0, 12)}'
                    ttl = rng.choice([None, 10, 50, 200])
                    if action == 'expired_save':
                        ttl = -rng.randrange(1, 20)
                    expires = None if ttl is None else utcnow() + timedelta(seconds=ttl)
                    generation = rng.randrange(0, 1_000_000)
                    response = make_response(
                        generation, url=f'https://contract.test/seq/{key}', expires=expires
                    )
                    cache.responses[key] = response
                    reference.save(key, cache.responses[key])
                    refresh_known_keys()

                elif action == 'overwrite' and known_keys:
                    key = rng.choice(known_keys)
                    response = make_response(
                        rng.randrange(0, 1_000_000),
                        url=f'https://contract.test/seq/{key}',
                        expires=utcnow() + timedelta(seconds=200),
                    )
                    cache.responses[key] = response
                    reference.save(key, cache.responses[key])

                elif action == 'delete' and known_keys:
                    n = rng.randrange(1, min(3, len(known_keys)) + 1)
                    keys = rng.sample(known_keys, n)
                    cache.delete(*keys)
                    reference.delete(*keys)
                    refresh_known_keys()

                elif action == 'bulk_delete' and known_keys:
                    n = rng.randrange(1, min(4, len(known_keys)) + 1)
                    keys = rng.sample(known_keys, n)
                    cache.responses.bulk_delete(keys)
                    reference.delete(*keys)
                    refresh_known_keys()

                elif action == 'advance':
                    shift = rng.choice([15, 45, 120])
                    traveller.shift(shift)
                    cache.delete(expired=True)
                    reference.remove_expired(utcnow())
                    refresh_known_keys()

                assert_contract(cache, reference)

            cache.close()

        reopened = reopen_cache(spec, tmp_path)
        try:
            if spec.persistent:
                assert_contract(reopened, reference)
            else:
                assert len(reopened.responses) == 0
        finally:
            reopened.close()


# --------------------------------------------------------------------------------------
# Concurrent last-committer semantics
# --------------------------------------------------------------------------------------


class TestConcurrencyContract:
    def test_concurrent_same_key_commit_order(self, cache, spec):
        """Two threads save different generations of the same key. Barriers pin the commit order:
        generation 1 reaches the commit point first and waits; generation 2 reaches it second;
        then both commit in that order (1 then 2).  Every backend with a defined write-ordering
        contract must retain generation 2 (last committer); at minimum the key must always resolve
        to a complete, readable body.
        """
        key = 'concurrent_key'
        errors = []
        original_setitem = type(cache.responses).__setitem__

        # Both threads wait until each has started its setitem, then gen1 waits for gen2 to arrive,
        # and gen2 waits for gen1's commit to finish. This fixes commit order as: gen1, then gen2.
        both_ready = threading.Barrier(2)
        gen1_at_commit = threading.Event()
        gen2_at_commit = threading.Event()
        gen1_committed = threading.Event()

        def ordered_setitem(self, k, value):
            if k != key:
                return original_setitem(self, k, value)
            generation = json.loads(bytes(value.content))['generation']
            both_ready.wait(5)
            if generation == 1:
                gen1_at_commit.set()
                assert gen2_at_commit.wait(5)
                result = original_setitem(self, k, value)
                gen1_committed.set()
                return result
            else:
                gen2_at_commit.set()
                assert gen1_at_commit.wait(5)
                # Gen2 commits strictly after gen1
                assert gen1_committed.wait(5)
                return original_setitem(self, k, value)

        type(cache.responses).__setitem__ = ordered_setitem
        threads = []
        try:
            def worker(generation):
                try:
                    cache.responses[key] = make_response(
                        generation, url='https://contract.test/cc'
                    )
                except Exception as e:  # pragma: no cover
                    errors.append(e)

            for generation in (1, 2):
                thread = threading.Thread(target=worker, args=(generation,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join(5)
        finally:
            type(cache.responses).__setitem__ = original_setitem

        assert not errors
        assert all(not thread.is_alive() for thread in threads)

        # Invariant for every backend: key resolves to a complete readable body
        assert key in cache.responses
        stored = cache.get_response(key)
        assert stored is not None
        assert stored.content
        generation = json.loads(stored.content)['generation']
        assert generation in (1, 2)

        # All backends tested here have an explicit last-committer-wins contract for
        # same-key writes, and commit order was pinned gen1 -> gen2.
        assert generation == 2


# --------------------------------------------------------------------------------------
# Serializer failures and corrupted bodies
# --------------------------------------------------------------------------------------


def _wrap_base_pickle(storage, *, fail_dumps=False, fail_loads=False):
    """Wrap the storage's existing pickle serializer so previously written bodies stay readable,
    while dumps/loads can independently be made to fail."""
    base = storage.serializer

    class PicklePassthrough:
        is_binary = True

        def dumps(self, value):
            if fail_dumps:
                raise RuntimeError('simulated encode failure mid-pipeline')
            return base.dumps(value) if hasattr(base, 'dumps') else pickle.dumps(value)

        def loads(self, value):
            if fail_loads:
                raise ValueError('simulated decode failure mid-pipeline')
            return base.loads(value) if hasattr(base, 'loads') else pickle.loads(value)

    storage.serializer = PicklePassthrough()


class TestSerializerFailureContract:
    @pytest.mark.parametrize('is_binary', [False, True])
    def test_encode_failure_leaves_no_orphan(self, spec, tmp_path, is_binary):
        """A failure while encoding a save must not leave a new key/index entry behind, and must
        not corrupt a body previously stored under the same key."""
        if not spec.corruptable:
            pytest.skip('memory backend stores objects directly, without an encode step')

        cache = fresh_cache(spec, tmp_path)
        try:
            with time_travel(EPOCH):
                good = make_response(1)
                cache.responses['existing'] = good

                # Existing bodies stay decodable; only new encodes fail mid-pipeline
                _wrap_base_pickle(cache.responses, fail_dumps=True)
                replacement = make_response(2)
                with pytest.raises(RuntimeError, match='simulated encode failure'):
                    cache.responses['brand_new'] = replacement
                with pytest.raises(RuntimeError, match='simulated encode failure'):
                    cache.responses['existing'] = replacement

                # No new record; no index pointing at a missing/partial body; old body intact
                assert 'brand_new' not in cache.responses
                assert_same_response(cache.get_response('existing'), good)
        finally:
            cache.close()


    def test_corrupted_body_is_invalid_and_pruned(self, spec, tmp_path):
        """If a stored body cannot be decoded, the backend treats the record as invalid instead of
        raising, and delete(invalid=True) removes it along with its dangling redirect."""
        if not spec.corruptable:
            pytest.skip('only backends with a serialized body layer can be corrupted')

        cache = fresh_cache(spec, tmp_path)
        try:
            with time_travel(EPOCH):
                cache.responses['good'] = make_response(1, url='https://contract.test/good')
                cache.responses['bad'] = make_response(2, url='https://contract.test/bad')
                cache.redirects['bad_alias'] = 'bad'
                _write_raw_body(cache, 'bad', b'this is not a valid serialized body')

                # Reads must fail soft: corrupted record resolves to None, not an exception
                assert cache.get_response('good') is not None
                assert cache.get_response('bad', default=None) is None

                cache.delete(invalid=True)
                assert 'bad' not in cache.responses
                assert 'bad_alias' not in cache.redirects
                assert 'good' in cache.responses
                assert len(cache.redirects) == 0
        finally:
            cache.close()

    def test_truncated_body_is_invalid_and_pruned(self, spec, tmp_path):
        """A truncated response body must not surface as a partial/corrupt object, and no index or
        redirect may point at the missing body."""
        if not spec.corruptable:
            pytest.skip('only backends with a serialized body layer can be truncated')

        cache = fresh_cache(spec, tmp_path)
        try:
            with time_travel(EPOCH):
                victim = make_response(42, url='https://contract.test/truncated')
                cache.responses['victim'] = victim
                cache.responses['survivor'] = make_response(43, url='https://contract.test/ok')
                cache.redirects['victim_alias'] = 'victim'

                raw = _read_raw_body(cache, 'victim')
                _write_raw_body(cache, 'victim', raw[: max(1, len(raw) // 2)])

                assert cache.get_response('victim', default=None) is None
                cache.delete(invalid=True)

                assert 'victim' not in cache.responses
                assert 'victim_alias' not in cache.redirects
                assert cache.get_response('survivor') is not None
                for alias in cache.redirects.keys():
                    assert cache.redirects[alias] in cache.responses
        finally:
            cache.close()


def _read_raw_body(cache: BaseCache, key: str) -> bytes:
    storage = cache.responses
    # SQLite backend
    if isinstance(cache, SQLiteCache):
        with storage.connection() as con:
            row = con.execute(
                f"SELECT value FROM {storage.table_name} WHERE key=?", (key,)
            ).fetchone()
        return row[0]
    # Filesystem backend
    if isinstance(cache, FileCache):
        return storage._key2path(key).read_bytes()
    raise AssertionError('raw body access is only defined for sqlite/filesystem')


def _write_raw_body(cache: BaseCache, key: str, body: bytes):
    storage = cache.responses
    if isinstance(cache, SQLiteCache):
        with storage.connection(commit=True) as con:
            con.execute(
                f"UPDATE {storage.table_name} SET value=? WHERE key=?",
                (sqlite3.Binary(body), key),
            )
    elif isinstance(cache, FileCache):
        path = storage._key2path(key)
        mode = 'wb' if storage.is_binary else 'wb'
        with path.open(mode) as f:
            f.write(body)
    else:
        raise AssertionError('raw body access is only defined for sqlite/filesystem')


# --------------------------------------------------------------------------------------
# Legacy fixture migration (SQLite only)
# --------------------------------------------------------------------------------------


SAMPLE_DATA_DIR = Path(__file__).resolve().parents[1] / 'sample_data'
LEGACY_FIXTURES = [
    SAMPLE_DATA_DIR / 'sample.db.1.0.0',
    SAMPLE_DATA_DIR / 'sample.db.1.2.0',
    SAMPLE_DATA_DIR / 'sample.db.1.3.0',
]


class TestLegacyMigrationContract:
    @pytest.mark.parametrize(
        'fixture',
        LEGACY_FIXTURES,
        ids=[path.name for path in LEGACY_FIXTURES],
    )
    def test_legacy_sqlite_fixture_opens_without_errors(self, fixture, tmp_path):
        """A cache file produced by an older requests-cache version must open cleanly: no
        dangling indexes/redirects, and new writes coexist with migrated data."""
        db_path = tmp_path / 'legacy.sqlite'
        db_path.write_bytes(fixture.read_bytes())

        cache = SQLiteCache(str(db_path))
        try:
            with time_travel(datetime(2099, 1, 1, tzinfo=timezone.utc)):
                # Every redirect alias stored in the legacy file must resolve to a present body
                # (or be pruned); nothing may point at a missing record.
                for alias in list(cache.redirects.keys()):
                    assert cache.redirects[alias] in cache.responses

                # Reading every legacy key must either return a response or be invalidated
                # gracefully (never raise).
                legacy_keys = list(cache.responses.keys())
                for key in legacy_keys:
                    result = cache.get_response(key, default=None)
                    assert result is None or result.status_code > 0

                # The migrated cache remains writable and new records round-trip correctly
                fresh = make_response(123, url='https://contract.test/after-migration')
                cache.responses['post_migration_key'] = fresh
                assert_same_response(cache.get_response('post_migration_key'), fresh)

                cache.delete(invalid=True, expired=True)
                for alias in cache.redirects.keys():
                    assert cache.redirects[alias] in cache.responses
        finally:
            cache.close()


# --------------------------------------------------------------------------------------
# Timezone-aware expiration
# --------------------------------------------------------------------------------------


class TestTimezoneAwareExpiration:
    @pytest.mark.parametrize(
        'expires',
        [
            pytest.param(EPOCH + timedelta(seconds=10), id='utc-aware'),
            pytest.param(
                EPOCH.astimezone(timezone(timedelta(hours=9))) + timedelta(seconds=10),
                id='tokyo-aware',
            ),
            pytest.param(
                datetime(2024, 1, 1, 0, 0, 10), id='naive-utc'),
        ],
    )
    def test_naive_and_aware_expiration(self, cache, expires):
        """Naive datetimes are treated as UTC; aware datetimes in other zones compare correctly."""
        key = 'tz_key'
        with time_travel(EPOCH) as traveller:
            response = make_response(1, url='https://contract.test/tz', expires=expires)
            cache.responses[key] = response

            assert not cache.get_response(key).is_expired
            traveller.shift(11)
            assert cache.get_response(key).is_expired
            cache.delete(expired=True)
            assert key not in cache.responses

    def test_aware_expiration_persists_across_reopen(self, spec, tmp_path):
        if not spec.persistent:
            pytest.skip('persistence across reopen only applies to persistent backends')
        cache = fresh_cache(spec, tmp_path)
        tokyo = timezone(timedelta(hours=9))
        try:
            with time_travel(EPOCH):
                # 09:00:10 in Tokyo == 00:00:10 UTC
                expires = datetime(2024, 1, 1, 9, 0, 10, tzinfo=tokyo)
                response = make_response(2, url='https://contract.test/tz-reopen',
                                         expires=expires)
                cache.responses['tz_key'] = response

            cache.close()
            reopened = reopen_cache(spec, tmp_path)
            try:
                with time_travel(EPOCH + timedelta(seconds=5)):
                    stored = reopened.get_response('tz_key')
                    assert stored is not None
                    assert not stored.is_expired

                with time_travel(EPOCH + timedelta(seconds=11)):
                    assert reopened.get_response('tz_key').is_expired
                    reopened.delete(expired=True)
                    assert 'tz_key' not in reopened.responses
            finally:
                reopened.close()
        finally:
            cache.close()
