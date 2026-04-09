from setuptools import setup

package_name = 'euclidean_clustering'

setup(
    name=package_name,
    version='0.0.0',
    packages=['lidarpipeline'],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/config', ['config/params.yaml', 'config/benchmark_results_bag.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='todo',
    maintainer_email='todo@example.com',
    description='Lidar ROI filtering and euclidean clustering with box, pizza-slice, and LUT filters.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_processor_node = lidarpipeline.lidar_processor_node:main',
        ],
    },
)
