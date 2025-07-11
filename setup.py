from setuptools import setup, find_packages

setup(
    name='multiagent_auction',
    version='0.1.0',
    packages=find_packages(),  # ou find_packages('src') se usar src-layout
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'imageio',
        'gym',
        'playsound'
    ],
    entry_points={
        'console_scripts': [
            'auction-sim = run:main',
        ]
    },
    python_requires='>=3.7',
    author='Seu Nome',
    description='Multi-agent auction simulations using DRL',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
