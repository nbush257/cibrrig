import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines() if not x.startswith("git+")]

setuptools.setup(
    name="cibrrig",
    version="0.1.0",
    author="Nicholas E. Bush",
    description="Tools for data analysis an dorganization of Neuropixel data recroded on the CIBR rig at SCRI. Influenced by the IBL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[],
    install_requires=require,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where=".", exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.8",
)