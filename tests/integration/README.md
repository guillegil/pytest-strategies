# Integration Tests

The tests in this folder run pytest-strategies the way a project uses it: through the
pytest plugin, with strategy files, fixtures, `conftest.py` hooks and command-line
options. The unit tests of each module are in `tests/unittests/`.

## Layout

| File                            | What it holds                                                                                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `strategies.py`                 | A strategy file: ten strategies registered with `@Strategy.register`, which the plugin imports before the test modules of this folder        |
| `test_full_integration.py`      | Tests decorated with `@Strategy.strategy(...)` that use those strategies, and tests that call `Parameter.generate_vectors()` themselves       |
| `test_fixtures_integration.py`  | Strategies used together with fixtures (see [FIXTURES_README.md](FIXTURES_README.md))                                                          |
| `test_integration.py`           | `RNG`, `TestArg` and `Parameter` working together, called directly                                                                             |
| `test_<feature>_integration.py` | One feature each, in a project that `pytester` writes and runs: test IDs, row streams, the context, records, the export, the repro output, ... |
| `test_fix_*_integration.py`     | Regression tests for the bugs that reviews found                                                                                               |
| `conftest.py`                   | The `values_dump` fixture (below)                                                                                                              |

### The strategies of `strategies.py`

- `simple_integer_strategy`: one integer argument, with the directed vectors `zero`,
  `fifty` and `max`.
- `multi_param_strategy`: an `int`, a `float` and a `bool`.
- `constrained_range_strategy` and `complex_constraint_strategy`: vector constraints,
  and RNG predicates (even numbers, multiples of 5).
- `api_request_strategy` and `database_query_strategy`: the parameters of an API
  request and of a database query, with vector constraints.
- `validated_strategy`: `TestArg` validators.
- `string_strategy`: `RNGString` and `RNGChoice`.
- `mixed_static_random_strategy`: a static value next to random ones.
- `fixed_rows_strategy`: rows that the factory computes from `nsamples`, given as
  directed vectors, with no random rows of its own (`Parameter(nsamples=0)`).

The factories take `nsamples` (see "Strategies and Factories" in the project
[README](../../README.md)).

## Running the tests

```bash
# Every integration test
pytest tests/integration -v

# The strategy-based tests
pytest tests/integration/test_full_integration.py -v

# Select rows by name: the directed vectors zero, zeros and all_zeros
pytest tests/integration/test_full_integration.py -k zero -v

# Leave out the random rows
pytest tests/integration/test_full_integration.py -k "not rand"

# Run one row again, with the values it had in the run with seed 42
pytest "tests/integration/test_full_integration.py::test_simple_integer[rand-3]" --rng-seed=42

# Change which rows exist
pytest tests/integration/test_full_integration.py --vector-mode=directed_only
pytest tests/integration/test_full_integration.py --nsamples=50
```

Each strategy row has a test ID that names it, the same for every seed:
`test_simple_integer[directed-zero]` for the directed vector `zero`,
`test_simple_integer[rand-0]` to `test_simple_integer[rand-9]` for the ten random rows
of the default `--nsamples`. A parametrized fixture puts its own ID first:
`test_strategy_with_parametrized_fixture[sqlite-directed-admin]`. The seed decides the
values of the random rows. It is printed in the report header, and when a row fails
the plugin prints the command that runs that row alone with the same values (see
"Reproducibility" in the project [README](../../README.md)).

### `test_fixed_rows`

`test_fixed_rows` gets its rows from directed vectors only, `row_0` to `row_9`, so the
vector mode treats them like any directed vectors:

- `--vector-mode=random_only` leaves the test without rows. With this project's
  `empty_parameter_set_mark = fail_at_collect`, the whole file then fails collection
  (`Empty parameter set in 'test_fixed_rows'`); add `-o empty_parameter_set_mark=skip`
  to skip that test instead.
- `--nsamples=N` gives it N directed rows plus N random rows of its static values,
  `(0, 0, 0)`, which also satisfy the test.

## Comparing runs

Node IDs name the rows, so they do not show whether two runs drew the same values.
The tests that compare runs compare the values instead, with the `values_dump`
fixture of `conftest.py`. Its `conftest` attribute holds a `conftest.py` for the
`pytester` project that writes the node ID and the parameter values of every
collected row, in collection order, to `values.json` (`values-<worker>.json` on a
pytest-xdist worker). `values_dump.collect(*args)` runs pytest with `args`, in a
subprocess by default, checks its exit code and returns those rows:

```python
def test_same_seed_same_values(pytester, values_dump):
    pytester.makeconftest(values_dump.conftest)
    pytester.makepyfile(test_x=TEST_MODULE)
    first = values_dump.collect("--collect-only", "--rng-seed=7")
    assert values_dump.collect("--collect-only", "--rng-seed=7") == first
```

`tests/golden/` holds the files that pin values and formats across releases:
`seed1.json` (the rows of `--rng-seed=1`, see `test_golden_values_integration.py`),
`export-schema1.json` (the export, see `test_export_schema_integration.py`) and the
test IDs of the names and values formats (see `test_names_ids_integration.py`).
