from setuptools import find_packages, setup

package_name = 'inspection_gui'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='col',
    maintainer_email='actonc@uw.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gui = inspection_gui.dynamic_pointcloud:main',
            'scanner = inspection_gui.scanner:main'
        ],
    },
)
