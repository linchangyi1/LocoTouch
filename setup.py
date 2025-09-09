from setuptools import setup, find_packages
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="LocoTouch",
    version="1.0.0",
    packages=find_packages(),
    install_requires=required,
    url="https://linchangyi1.github.io/LocoTouch/",
    author="Changyi Lin",
    author_email="changyil@andrew.cmu.edu",
    description="Open source of LocoTouch.",
    python_requires=">=3.10",
)
