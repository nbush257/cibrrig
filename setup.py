import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines() if not x.startswith("git+")]

setuptools.setup(
    name="cibrrig",
    version="0.2.9",
    author="Nicholas E. Bush",
    description="Tools for data analysis and organization of Neuropixel data recroded on the CIBR rig at SCRI. Influenced by the IBL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=require,
    packages=setuptools.find_packages(exclude=['_wirings','.vscode']),
    include_package_data=True,
    python_requires=">=3.8",
    entry_points={
        "console_scripts":[
            "backup = cibrrig.archiving.backup:main",
            "npx_preproc = cibrrig.preprocess.preproc_pipeline:main",
            "npx_run_all = cibrrig.main_pipeline:main",
            "ephys_to_alf = cibrrig.archiving.ephys_data_to_alf:main",
            "spikesort = cibrrig.sorting.spikeinterface_ks4:main",
        ]
    },
)