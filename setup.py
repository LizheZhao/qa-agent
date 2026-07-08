from setuptools import setup, find_packages

setup(
    name="flake8-custom-rules",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "flake8.extension": [
            "G001 = plugins.flake8_custom_rules:NoPrintChecker",
        ],
    },
)
