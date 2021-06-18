import setuptools


setuptools.setup(
    name="iba",
    url="https://github.com/BioroboticsLab/IBA",
    version="0.0.1",
    author="Karl Schulz, Leon Sixt",
    author_email="karl.schulz@tum.de, leon.sixt@fu-berlin.de",
    license='MIT',
    description="Information Bottlenecks for Attribution (IBA)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy', 
        'scikit-image', 
        'tqdm', 
        'Pillow',
        'packaging',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov',
            'pytest-readme',
            'flake8',
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx-autobuild',
        ],
        'torch': [
            'torch>=1.1.0',
            'torchvision>=0.3.0',
        ],
        'tensorflow-v1': [
            'tensorflow>=1.12.0, <2.0',
            'tensorflow-probability<=0.7.0',
            'keras<2.3.0',
        ],
        'tensorflow-gpu': [
            'tensorflow-gpu>=1.12.0, <2.0'
            'keras<2.3.0',
        ],
    },
    python_requires='>=3.6',
    keywords=['Deep Learning', 'Attribution', 'XAI'],
)
