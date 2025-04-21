from setuptools import find_packages, setup

package_name = 'autoware_ml_motion_prediction'

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
    maintainer='ahsan',
    maintainer_email='ahmedahsan155@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hello_world_node = autoware_ml_motion_prediction.nodes.hello_world_node:main'
        ],
    },
)
