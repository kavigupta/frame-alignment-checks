import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frame-alignment-checks",
    version="0.0.23",
    author="Kavi Gupta",
    author_email="frame-alignment-checks@kavigupta.org",
    description="Library for determining whether a RNA splicing predictor is using frame alignment information",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kavigupta/frame-alignment-checks",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"frame_alignment_checks.data": ["**/*.npz", "**/*.pkl", "**/*.xlsx", "**/*.gz"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[""],
)
