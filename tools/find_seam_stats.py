from __future__ import print_function, division

import includes

import sys
import obj_reader
import find_seam

mesh = obj_reader.load_obj(sys.argv[1])
print("#Vertices:", len(mesh.v))
print("#Faces:", len(mesh.f))

seams, boundaries, foldovers = find_seam.find_seam(mesh)

print("seam edges:")
# print(seams)
print(len(seams))
# for seam in seams:
#     print(seam[0][0], seam[0][1][0], seam[1][0], seam[1][1][0])

print("boundary edges:")
print(len(boundaries))
# for seam in boundaries:
#     print(seam[0], seam[1][0])

print("foldover edges:")
print(len(foldovers))
# for seam in foldovers:
#     print(seam[0][0], seam[0][1][0], seam[1][0], seam[1][1][0])
# for seam in foldovers:
#     print(seam[0], seam[1][0])
