from setuptools import setup, find_packages

setup(
    name='geak_eval',  # Replace with your module/package name
    version='0.1.0',
    author='Vinay Joshi',
    author_email='vinajosh@amd.com',
    description='An evaluation framework for Triton kernels',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add your dependencies here
        # 'requests>=2.0.0',
    ],
    entry_points={
        'console_scripts': [
            'geak-eval=geak_eval.run:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
