from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='MotionArtifactRemoval',
    version='1.0.0',    
    description='Motion Artifact Removal',
    author='Jim Peterson, Abed Ghanbari',
    url='https://www.jax.org',
    author_email='abed.ghanbari@jax.org',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=required,

    classifiers=[
    ],
    include_package_data=True,
    package_data={
        'MotionArtifactRemoval.msUNET': ['predict/scripts/trained_models/*.hdf5'], 
        'MotionArtifactRemoval.motion_detector': ['trained_models/*.pkl'], 
                },
)
