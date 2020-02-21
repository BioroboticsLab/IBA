import setuptools

setuptools.setup(
    name="iba",
    version="0.0.1",
    author="Karl Schulz, Leon Sixt",
    author_email="",
    license='MIT',
    description="Implementation of the Attribution Bottleneck",
    packages=setuptools.find_namespace_packages(include=['iba.*']),
    install_requires=['numpy', 'scikit-image'],
    python_requires='>=3.6',
    keywords=['Deep Learning', 'Attribution'],
)
