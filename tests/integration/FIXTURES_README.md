# Strategies and Fixtures

`test_fixtures_integration.py` checks that strategies and pytest fixtures work
together: a strategy row and any fixtures in one test, with no configuration. This
page says how the two combine; the project [README](../../README.md) has the full
rules.

## How a test gets its arguments

`@strategy("name")` parametrizes the test with the strategy's argument names. Every
other parameter of the test is a fixture, which pytest provides as usual:

```python
@pytest.fixture
def sample_config() -> dict[str, int]:
    return {"max_retries": 3}


@Strategy.strategy("fixture_compatible_strategy")
def test_strategy_with_single_fixture(user_id: int, operation: str, sample_config: dict[str, int]):
    assert 1 <= user_id <= 1000  # user_id and operation come from the strategy
    assert sample_config["max_retries"] == 3  # sample_config from the fixture
```

- Built-in fixtures (`request`, `tmp_path`, `monkeypatch`, `capsys`, ...) and your
  own fixtures, with setup and teardown or with any scope, work the same way.
- A strategy argument that has the name of a fixture replaces that fixture in the
  test, as `@pytest.mark.parametrize` does.
- With `validate_signature=True`, the default, every strategy argument must be asked
  for, by the test or by a fixture it uses; otherwise collection fails:

  ```text
  In test_missing: Signature validation failed for strategy 'users': Test function signature mismatch for strategy 'users'!
    Strategy provides: ['user_id', 'role']
    Test function expects: ['user_id']
    Missing parameters: ['role']
  ```

## Fixtures that ask for strategy arguments

A fixture can take strategy arguments itself, and build what the test needs from each
row. The test then asks for the fixture only:

```python
# conftest.py
@pytest.fixture
def user(user_id, role):
    return {"id": user_id, "role": role}


# test_users.py
@strategy("users")
def test_login(user):
    assert user["role"] in ("admin", "user", "guest")
```

A test can also receive the whole row as one record, a dataclass whose fields are the
strategy's arguments. The rule: **a test receives the row as one record when (1)
neither the test nor any fixture it uses asks for one of the strategy's argument
names, and (2) exactly one test parameter is annotated with a record type whose
fields are exactly those names. Otherwise the strategy passes its arguments by name,
one test parameter each.** So a fixture like `user` above makes the strategy pass the
arguments by name, and a test that also takes a record of them fails collection (see
"Record Parameters" in the README).

## Parametrized fixtures and test IDs

A parametrized fixture multiplies the rows: each strategy row runs with each of the
fixture's parameters. pytest puts the fixture's ID before the strategy's:

```python
@pytest.fixture(params=["sqlite", "postgres", "mysql"])
def database_type(request):
    return request.param


@Strategy.strategy("fixture_compatible_strategy")
def test_strategy_with_parametrized_fixture(user_id: int, operation: str, database_type: str):
    assert database_type in ["sqlite", "postgres", "mysql"]
```

```text
test_strategy_with_parametrized_fixture[sqlite-directed-admin]
test_strategy_with_parametrized_fixture[sqlite-directed-user]
test_strategy_with_parametrized_fixture[sqlite-rand-0]
...
test_strategy_with_parametrized_fixture[mysql-rand-9]
```

The IDs name the rows, so they stay the same for every seed, and `-k` selects by both
parts:

```bash
pytest tests/integration/test_fixtures_integration.py -k "parametrized and sqlite and directed"
pytest "tests/integration/test_fixtures_integration.py::test_strategy_with_parametrized_fixture[mysql-rand-3]" --rng-seed=42
```

## The row and the context in fixtures

- The item of each strategy row carries the row's `VectorInfo` in
  `request.node.stash[VECTOR_KEY]` (its strategy, kind, name, index, values and
  seed), for fixtures that log or label the row. Read it with
  `request.node.stash.get(VECTOR_KEY, None)` in a fixture that also serves tests
  without a strategy.
- The `strategies_ctx` session fixture gives the object that the
  `pytest_strategies_context` hook returned for the tests that use it, the one the
  factories received, and `pytest_strategy.get_context(config, path)` gives the
  context of a folder. A testbench fixture can build on it instead of reading the
  configuration again (see "The context in fixtures" in the README).

## Random draws in fixtures

`RNG` draws made in a fixture follow `--rng-seed`. Each fixture's setup draws from a
stream derived from the seed, the node ID the fixture is set up for, its name, its
parameter index and where it is defined. So a fixture draws the same values whether
its test runs alone, in the suite, in another order or under pytest-xdist:

```python
@pytest.fixture
def token():
    return RNG.integer(0, 2**32 - 1)
```

So `pytest "test_users.py::test_token[rand-1]" --rng-seed=1` gives the `token` that
row had in a whole run with `--rng-seed=1`. Python's own `random` module does not
follow the seed (see "Draws outside the rows" in the README).

## The tests

| Test                                     | Fixtures                                                                    |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| `test_strategy_with_single_fixture`      | `sample_config`                                                             |
| `test_strategy_with_multiple_fixtures`   | `sample_config`, `api_client`, `database_connection`                        |
| `test_api_with_client_fixture`           | `api_client`                                                                |
| `test_strategy_with_cleanup_fixture`     | `resource_with_cleanup`, with teardown                                      |
| `test_strategy_with_parametrized_fixture` | `database_type`, three parameters                                           |
| `test_strategy_with_request_fixture`     | `request`                                                                   |
| `test_strategy_with_tmp_path`            | `tmp_path`                                                                  |
| `test_strategy_with_monkeypatch`         | `monkeypatch`                                                               |
| `test_strategy_with_capsys`              | `capsys`                                                                    |
| `test_complex_integration`               | `api_client`, `database_connection`, `sample_config`, `tmp_path`, `monkeypatch` |

The strategies `fixture_compatible_strategy` and `api_test_strategy` are registered
in the test module itself. Fixtures that ask for strategy arguments, records,
`strategies_ctx` and fixture streams have their own tests in
`test_records_integration.py`, `test_strategies_ctx_integration.py` and
`test_plugin_streams_integration.py`.
