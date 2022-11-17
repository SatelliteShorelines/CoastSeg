# A setup.py to read markdown on readme and render it correctly on the pypi page
from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()
    
setup(
    long_description=readme,
    long_description_content_type="text/markdown",
)