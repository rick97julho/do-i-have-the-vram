from setuptools import setup, find_packages

setup(
    name="do-i-have-the-vram",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "huggingface_hub>=0.19.0",
        "tqdm>=4.66.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "do-i-have-the-vram=do_i_have_the_vram.cli:main",
        ],
    },
)
