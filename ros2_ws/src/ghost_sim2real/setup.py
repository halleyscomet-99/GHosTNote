from setuptools import setup

package = "ghost_sim2real"
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
    description="GHOST sim2real runs (goto, strokes, k sweeps, MIDI scoring) and the MuJoCo twin viewer and the kit in RViz on ROS 2.",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "run = ghost_sim2real.run:main",
        "twin_viewer = ghost_sim2real.twin_viewer:main",
        "kit_markers = ghost_sim2real.kit_markers:main",
    ]},
)
