import setuptools

setuptools.setup(
    name="per-sample-bottleneck",
    version="0.0.1",
    author="Anonymous",
    author_email="",
    description="Implementation of the Attribution Bottleneck",
    packages=setuptools.find_namespace_packages(include=['per_sample_bottleneck.*']),
    install_requires=['torch'],
    python_requires='>=3.6',
)
