# Issues

Please use GitHub Issues to report bugs or suggest features. 
Include enough context (steps to reproduce, expected vs. actual behavior) so we can reproduce the issues and help you effectively.

# Pull Requests

## Steps

Workflow
- Fork the repository and create your branch from `main`
- Make your code changes
- Test your code changes locally, see the section below
- Push the changes to your fork
- Open a Pull Request ONLY if the tests pass

If the testing procedure seems too difficult for you or you can't get the tests to pass, you can still open an Issue instead and reference your changes there.

## Testing

Details on the testing procedure are listed [here](https://drocheam.github.io/optrace/development/testing.html).
To minimize the chance of committing erroneous features, testing should be done locally first.
Depending on your changes, the following testing should be performed:

| Change Area  | Command                           |
| :----------- | :-------------------------------- |
| `optrace/*`  | `tox` and `tox -e docs`           |
| `examples/*` | `pytest ./tests/test_examples.py` |
| `tests/*`    | `pytest [affected_test_file]`     |
| `docs/*`     | `tox -e docsbuildcheck`           |

# License

By contributing to this repository, you agree that your code is your own work and will be licensed under MIT license provided in the [LICENSE](./LICENSE) file.



