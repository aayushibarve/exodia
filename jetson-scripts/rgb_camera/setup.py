from setuptools import find_packages, setup

package_name = 'rgb_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' +package_name, ['launch/rgb_thermal_imu.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='disal',
    maintainer_email='aayushibarve06@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'rgb_camera_node = rgb_camera.rgb_camera_node:main',
        ],
    },
)
