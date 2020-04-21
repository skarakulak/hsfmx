from setuptools import setup
setup(
        name='hsfmx',
        version='0.1',
        description='Efficient Ensemble of Hierarchical Softmaxes for a High-Rank Language Model',
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="NLP deep learning hierarchical softmax pytorch",
        author='Serkan Karakulak',
        author_email='serkankarakulak@gmail.com',
        license='MIT',
        url='http://github.com/skarakulak/hsfmx',
        packages=['hsfmx'],
        zip_safe=False,
        install_requires=['numpy'],
        python_requires=">=3.6.0",
        )
