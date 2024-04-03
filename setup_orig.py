import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="MAVImysetup", # Replace with your own username
    version="2024.2.26",
    author="Kiro",
    # author_email="jure.brence@ijs.si",
    # description="Probabilistic grammar-based equation discovery",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="https://github.com/brencej/ProGED",
    packages=setuptools.find_packages(),
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    python_requires='>=3.6',
    install_requires = ["numpy", 
                        # "pandas", 
                        # "scipy", 
                        # "sympy", 
                        # "nltk",
                        # "scikit-learn",
                        # "hyperopt",
                        # "diophantine",
                        # "pytest",
                       ]
)
