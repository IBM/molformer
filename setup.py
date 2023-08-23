from setuptools import setup, find_packages

setup(
    name='molformer',
    version='0.1.0',
    packages=find_packages(include=['src/molformer', 'src/molformer.*']),
    url='https://github.com/lamalab-org/molformer',
    license='MIT',
    author='Kevin Maik Jablonka and Adrian Mirza',
    author_email='',
    description='Refactored MolFormer',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    package_data={'src/molformer': ['data']},
    include_package_data=True
    install_requires=[
        'transformers==4.6.0',
        'pytorch-lightning==1.1.5',
        'pytorch-fast-transformers==0.4.0',
        'datasets==1.6.2',
        'jupyterlab==3.4.0',
        'ipywidgets==7.7.0',
        'bertviz==1.4.0'
    ]
    )