## PEP-8 Coding Style 
- Reference: https://peps.python.org/pep-0008/

## Enforcement Scope
- `test-*`: no check (for local development) 
- `src/model/*`: `print()` is prohibited
- Other files: check on flake8 `N,E,F,C,B` code is enforced

## How to Coding Style Check Locally?
- `pip install -r requirements.txt`
- `pip install .`
- Run Flake8: `flake8 .`

You will get output similar to this if there are coding style violations

![flake8-output](/assets/flake8-output.png)

To opt out style check for local development,  in `.flake8` file
- Add custom exclusion rules to `per-file-ignores = `
- Exclude your venv by adding path to `exclude = `

Currently, all files of pattern `test-*.py` are excluded from the style check. 