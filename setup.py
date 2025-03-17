from setuptools import setup

setup(
    name='MultiFingerTrajOpt',
    version='0.1.0',
    packages=['optimization'],
    author='fanyang',
    author_email='fanyangr@umich.edu',
    description='MultiFingered Trajectory Optimization for Allegro Hand',
    install_requires=[
        'torch',
        'numpy',
    ],
    tests_require=[
        'pytest'
    ]
)
