from setuptools import setup

package_name = "navsim3d_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="navsim3d",
    maintainer_email="navsim3d@example.com",
    description="ROS 2 wrapper for navsim3d swarm visualization.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "swarm_node = navsim3d_ros.swarm_node:main",
        ],
    },
)
