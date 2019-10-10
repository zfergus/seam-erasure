# Seam Erasure

[![Build Status](https://travis-ci.com/zfergus/seam-erasure.svg?branch=master)](https://travis-ci.com/zfergus/seam-erasure)

<img src = "static/img/teaser.png" width="100%">

**Seamlessly erase seams from your favorite 3D models.**

* [Project Page](https://cragl.cs.gmu.edu/seamless/)
* [Paper](https://goo.gl/1LwB3Z)
* [Video](https://youtu.be/kCryf9n82Y8)
* [Seam Aware Decimation Code](https://github.com/songrun/SeamAwareDecimater)

Erases texture seams to prevent visible seams or tearing in various texture
maps (color, normal, displacement, ambient occlusion, etc.).

## Overview

Textures seams often produce errors when bi-linearly interpolated. This
results in a visible seam line or other undesired artifacts. The goal of this
project is to devise a numerical solution to this problem by minimizing the
energy/error between interpolated values of edge pairs.

## Dependencies

Install all required dependencies using: `pip install -r requirements.txt`

Dependencies include:
* `numpy`: linear algebra
* `scipy`: sparse matrix operations
* `recordclass`: simple data objects
* `pillow`: saving/loading texture image files

## Usage

### Command-Line

To use the seam eraser tool, first install the required packages, then
enter the following into a command-line:

```bash
python SeamErasure/main.py path/to/input_model path/to/input_texture [-h] [-o path/to/output_texture] [-g] [--sv {none,texture,lerp}] [-d]
```

Positional arguments:
* `path/to/input_model`: Path to input mesh file.
* `path/to/input_texture`: Path to input texture image or directory to load all textures from.

Optional arguments:
* `-o path/to/output_texture`, `--output path/to/output_texture`: Name of output texture or directory to save batch textures.
* `-g`, `--global`: Should the minimization have global effects? (default: False)
  * This should be used if the texture has global discontinuities. This will propagate changes from the seam inward (see the teaser image for a example of global discontinties).
* `--sv {none,texture,lerp}`: What method should be used to compute the seam value
energy? (default: `none`)
    * `none`: do not use a seam value term
    * `texture`: use difference in original texture
    * `lerp`: use linearly interpolated values along the edge
        * Values are provided at the vertices in the `.obj` as additional entries after the xyz triplet (e.g. `v <x> <y> <z> [<r>] [<g>] [<b>] [<a>] ...` where the additional channels are optional and must match the number of channels in the texture image).
* `-d`, `--data`: Should the input texture(s) be loaded as a `.data` files? (default: False)

## Files

* `server.py`: Flask based Python code for handling web inputs.
* `SeamErasure/`: Python package for Seam-Erasure
    * `SeamErasure/main.py`: Command-line interface for seam erasure.
* `examples/`: examples 3D model and texture files for testing
* `static/`: Static web page content including style sheets
* `templates/`: HTML template pages

### Web Browser UI

This repository also includes a Flask based server implementation that can be
run locally. This provides a simple webpage interface to provide model/texture
input and select options. To get this user interface run:

```
python server.py
```

This will start a server on the localhost. Navigate to the outputted address
in your choice of web browser to view the interface.

## Results

### Diffuse Textures

| Before | After |
|:------:|:-----:|
| <img src="static/img/results/Diffuse-Textures/cow-horn-before.png">    | <img src="static/img/results/Diffuse-Textures/cow-horn-after.png">    |
| <img src="static/img/results/Diffuse-Textures/chimp-hand-before.png">     | <img src="static/img/results/Diffuse-Textures/chimp-hand-after.png">     |
| <img src="static/img/results/Diffuse-Textures/teapot-red-before.png">    | <img src="static/img/results/Diffuse-Textures/teapot-red-after-global.png">    |

### Normal Maps

| Before | After |
|:------:|:-----:|
| <img src="static/img/results/Normal-Map-Results/cow/cow-horn-nm-before.png"> | <img src="static/img/results/Normal-Map-Results/cow/cow-horn-nm-after.png"> |
| <img src="static/img/results/Normal-Map-Results/cow/neck-before.png">  | <img src="static/img/results/Normal-Map-Results/cow/neck-after.png">  |
| <img src="static/img/results/Normal-Map-Results/lemon/lemon-before.png">  | <img src="static/img/results/Normal-Map-Results/lemon/lemon-after.png">  |
| <img src="static/img/results/Environment-Map/lemon-tilt-desert-before2.png">  | <img src="static/img/results/Environment-Map/lemon-tilt-desert-after2.png">  |

### Ambient Occlusion

| Before | After |
|:------:|:-----:|
| <img src="static/img/results/Ambient-Occlusion/hercules-before.png"> | <img src="static/img/results/Ambient-Occlusion/hercules-after.png"> |

### Geometry Images

| Before | After |
|:------:|:-----:|
| <img src="static/img/results/Geometry-Images/lemon-before.png"> | <img src="static/img/results/Geometry-Images/lemon-after.png"> |
| <img src="static/img/results/Geometry-Images/lemon-side-before.png">  | <img src="static/img/results/Geometry-Images/lemon-side-after.png">  |
