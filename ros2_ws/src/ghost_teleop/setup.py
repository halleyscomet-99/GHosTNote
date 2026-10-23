from setuptools import setup

package = "ghost_teleop"
setup(
    name=package,
    version="0.1.0",
    packages=[package],
    data_files=[("share/ament_index/resource_index/packages", [f"resource/{package}"]),
                (f"share/{package}", ["package.xml"])],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Halley Zhong",
    maintainer_email="halleyscomet678@gmail.com",
    description="GHOST teleop: Myo EMG stiffness and IMU arm motion into ghost_controller.",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "myo_bridge = ghost_teleop.myo_bridge:main",
        "imu_teleop = ghost_teleop.imu_teleop:main",
    ]},
)
