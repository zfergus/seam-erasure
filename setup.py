"""Install seam_erasure."""

from distutils.core import setup

import seam_erasure

setup(
    name=seam_erasure.__name__,
    packages=[seam_erasure.__name__],
    version=seam_erasure.__version__,
    license=seam_erasure.__license__,
    description="Seamlessly erase seams from your favorite 3D models.",
    long_description=(
        "Erases texture seams to prevent visible seams or tearing in various "
        "texture maps (color, normal, displacement, ambient occlusion, etc)."),
    author=seam_erasure.__author__,
    author_email=seam_erasure.__email__,
    url="https://github.com/zfergus/seam-erasure",
    download_url="https://github.com/zfergus/seam-erasure/archive/v0.1.tar.gz",
    # Keywords that define your package best
    keywords=["3D Modeling", "Textures", "Computer Graphics"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "License :: OSI Approved :: MIT License",
        # Specify which pyhton versions that you want to support
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires=">= 2.7",
    install_requires=[
        "numpy",
        "scipy",
        "recordclass",
        "pillow",
        "pathlib; python_version < '3.4'",
        "tqdm",
    ],
    extras_require={
        "cholmod": ["cvxopt"],
        "web-ui": ["flask"],
    },
    entry_points={  # Optional
        "console_scripts": [
            "seam-erasure=main:main",
            "seam-erasure-webui=server:main",
        ],
    },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/zfergus/seam-erasure/issues",
        "Research Project Page": "https://cragl.cs.gmu.edu/seamless/",
        "Paper": "https://goo.gl/1LwB3Z",
        "Video": "https://youtu.be/kCryf9n82Y8",
        "Source": "https://github.com/zfergus/seam-erasure/",
    },
)
