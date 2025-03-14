from setuptools import setup

setup(
    name='ccai',
    version='0.1.0',
    packages=['ccai'],
    author='zhsh',
    author_email='tpower@umich.edu',
    description='Constrained control-as-inference with SVGD',
    install_requires=[
        'torch',
        'numpy',
    ],
    tests_require=[
        'pytest'
    ]
)
