Quick Start
===========

.. code::

    from recursive_diff import recursive_diff

    lhs = {
        'foo': [1, 2, ('one', 5.2), 4],
        'only_lhs': 1
    }
    rhs = {
        'foo': [1, 2, ['two', 5.200001, 3]],
        'only_rhs': 1
    }

    for diff in recursive_diff(lhs, rhs, abs_tol=.1):
        print(diff)

Output::

    Pair only_lhs:1 is in LHS only
    Pair only_rhs:1 is in RHS only
    [foo]: LHS has 1 more elements than RHS: [4]
    [foo][2]: object type differs: tuple != list
    [foo][2]: RHS has 1 more elements than LHS: [3]
    [foo][2][0]: one != two


Or as a unit test:

.. code::

    from recursive_diff import recursive_eq

    def test1():
        recursive_eq(lhs, rhs, abs_tol=.1)

py.test output::

    ==================== FAILURES ===================
    E       AssertionError: 6 differences found

    -------------- Captured stdout call --------------

    Pair only_lhs:1 is in LHS only
    Pair only_rhs:1 is in RHS only
    [foo]: LHS has 1 more elements than RHS: [4]
    [foo][2]: object type differs: tuple != list
    [foo][2]: RHS has 1 more elements than LHS: [3]
    [foo][2][0]: one != two


Compare two nested directory trees that contain ``.json``, ``.jsonl``, ``.yaml``,
``.msgpack``, ``.nc``, or ``.zarr`` files:

.. code::

    from recursive_diff import recursive_open, recursive_eq

    lhs = recursive_open("baseline")
    rhs = recursive_open("new_output")
    recursive_eq(lhs, rhs)


Same as above, but from the command line::

    $ recursive-diff -r baseline new_output
