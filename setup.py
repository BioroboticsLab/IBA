import setuptools

setuptools.setup(
    name="iba",
    version="0.1.0",
    author="Karl Schulz, Leon Sixt",
    author_email="",
    license='MIT',
    description="Information Bottlenecks for Attribution (IBA)",
    packages=setuptools.find_namespace_packages(include=['IBA.*']),
    install_requires=['numpy', 'scikit-image', 'tqdm', 'pytest'],
    extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov',
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx-autobuild',
        ],
        'torch': [
            'torch',
            'torchvision',
        ],
        'tensorflow': [
            'tensorflow>=1.12.0, <2.0',
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
