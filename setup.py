from setuptools import setup, find_packages

setup(
    name="molformer",
    version="0.1.0",
    packages=find_packages(
        where="src",
        include=["molformer"],
    ),
    package_dir={"": "src"},
    url="https://github.com/lamalab-org/molformer",
    license="MIT",
    author="Adrian Mirza and Kevin Maik Jablonka",
    author_email="",
    description="Refactored MolFormer",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=[
        "transformers==4.31.0",
        "lightning==2.0.8",
        "pytorch-fast-transformers==0.4.0",
        "datasets==1.6.2",
        "jupyterlab==3.4.0",
        "ipywidgets==7.7.0",
        "bertviz==1.4.0",
    ],
)
