# Seam Erasure

<img src = "static/img/teaser.png" width="100%">

[http://cragl.cs.gmu.edu/seamless](http://cragl.cs.gmu.edu/seamless)

Erases texture seams to prevent visible seams or tearing in various texture maps (color, normal, displacement, ambient occlusion, etc.).

## Overview

Seams of textures often produce errors when bi-linearly interpolated. This
results in a visible seam line or other undesired artifacts. The goal of this
project is to devise a numerical solution to this problem by minimizing the
energy/error between edge pairs.

This repository contains a release candidate for the development version found
[here](http://github.com/zfergus/seam-erasure-dev). See the Seam-Erasure-Dev
repository for the latest and experimental developments.

## Dependencies

This repository is designed for Python 2, but it should be compatible with
Python 3 as well.

Python libraries: (`sudo pip[3] install <package-name>`)
* recordclass: simple data objects
* scipy: sparse matrix operations
* numpy: linear algebra
* pillow: saving/loading texture image files

### Installation

Install for Python 2.x/3.x:

```bash
sudo apt-get install python[3]-dev python[3]-pip
sudo -H pip[3] install numpy scipy pillow recordclass
```

## Usage

To use the seam eraser tool, first install the required packages, then
enter the following into a cmd-line:

```bash
python ./src/main.py path/to/input_model path/to/input_texture [-h] [-o path/to/output_texture] [-g] [--sv {none,texture,lerp}] [-d]
```
Positional arguments:
* `path/to/input_model`: Path to input mesh file.
* `path/to/input_texture`: Path to input texture image or directory to load all textures from.

Optional arguments:
* `-h`, `--help`: Show this help message and exit
* `-o path/to/output_texture`, `--output path/to/output_texture`: Name of output texture or directory to save batch textures.
* `-g`, `--global`: Should the minimization have global effects? (default: False)
  * **This should be used if the texture has global discontinies. This will propogate changes from the seam inward. See the teaser image for a example of global discontinties (the texture is a gradient image).**
* `--sv {none,texture,lerp}`What method should be used to compute the seam value
energy? None implies do not use seam value. Texture implies use difference in originial texture. Lerp
implies use linearly interpolated values along the edge. (default: none)
* `-d`, `--data`: Should the input texture(s) be loaded as data files? (default: False)
                        
## Files

* `server.py`: Flask based Python code for handling web inputs.
* `SeamErasure/`: Python package for Seam-Erasure
    * `SeamErasure/main.py`: Command-line interface for seam erasure.
* `static/`: Static web page content including style sheets
* `templates/`: HTML template pages

## Results

### Diffuse Textures

| Before | After |
|:------:|:-----:|
| <img src="static/img/results/Diffuse-Textures/cow-horn-before.png">    | <img src="static/img/results/Diffuse-Textures/cow-horn-after.png">    |
| <img src="static/img/results/Diffuse-Textures/chimp-hand-before.png">     | <img src="static/img/results/Diffuse-Textures/chimp-hand-after.png">     |
| <img src="static/img/results/Diffuse-Textures/nefertiti-before.png">    | <img src="static/img/results/Diffuse-Textures/nefertiti-after.png">    |
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
