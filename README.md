# Seam Minimization Web Service
Energy minimization of texture seams to prevent visible seams or tearing in
displacement maps.

## About

Seams of a textures often produce errors when bi-linearly interpolated. This
results in a visible seam line or other undesired artifacts. The goal of this
project is to devise a numerical solution to this problem by minimizing the
energy/error between edge pairs.

This repository contains a server based web service version of
[Seam-Minimization](http://github.com/zfergus2/Seam-Minimization). See the
Seam-Minimization repository for the command-line tool, and the latest
developments.

## Files

* `server.py` - Flask based python code for handling web inputs.
* `SeamMin/` - Python package for Seam-Minimization
* `static/` - Static web page content including style sheets
* `templates/` - HTML template pages
