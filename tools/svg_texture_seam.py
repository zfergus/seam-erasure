"""
Visualization tool for creating a SVG of an OBJ textures's seam.
Written by Zachary Ferguson
"""

from __future__ import print_function, division

import os
import sys
import getopt
import random

import includes

import obj_reader
from find_seam import find_seam, seam_to_UV
from seam_loops import find_seam_loops, find_edge_loops


def save_lines(outFilename, lines):
    """
    Save out lines to the a file with the given outFilename. Appends a newline
    after each element of lines.
    """
    print("Saving:", outFilename)
    with open(outFilename, 'w') as out:
        for line in lines:
            print(line, file = out)


def generate_svg_seam(mesh, width = 1024, height = 1024,
        styleFormatStr = "stroke:#%06x;stroke-width:8;"):
    """
    Genererate the SVG lines for the given models UV seam.
    Inputs:
        mesh   - 3D model to generate UV faces. Should match the specifications
            of OBJ recordclass.
        width  - width of output SVG texture (Default: 1024)
        height - height of output SVG texture (Default: 1024)
        styleFormatStr - Valid SVG style string for the style of the edges.
            Should have a %06x format string for the stroke color.
            (Default: "stroke:#%06x;stroke-width:8;")
    Output:
        Returns a list of the lines for the SVG.
    """

    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" ' +
        '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">')
    lines.append('<svg xmlns="http://www.w3.org/2000/svg" ' +
        'height = "%d" width = "%d">' % (width, height))
    lines.append(
        '<rect width="%d" height="%d" style="fill:black;stroke-width:0;"/>' %
        (width, height)
    )

    print("\nFinding seam of model")
    seam, boundaries, foldovers = seam_to_UV(mesh, *find_seam(mesh))
    print("Done\n")

    # style = styleFormatStr % int(random.random() * 0xFFFFFF)
    # def edges_to_lines(edges, style):
    #     return ['<line x1="%g" y1="%g" x2="%g" y2="%g" style="%s" />' %
    #         (edge[0].u * width, height - (edge[0].v * height),
    #          edge[1].u * width, height - (edge[1].v * height), style)
    #         for edge in edges]
    #
    # lines += edges_to_lines([edge for edgePair in seam for edge in edgePair],
    #     styleFormatStr % 0xFF0000)
    # lines += edges_to_lines(boundaries, styleFormatStr % 0x00FF00)
    # lines += edges_to_lines(foldovers, styleFormatStr % 0x0000FF)

    def edge_loops_to_polylines(loops, style):
        li_points = []
        for loop in loops:
            if(len(loop) == 0):
                continue
            loop = [((edge[0].u * width, height - (edge[0].v * height)),
                     (edge[1].u * width, height - (edge[1].v * height)))
                    for edge in loop]
            # import pdb
            # pdb.set_trace()
            tmp = [("%g,%g" % loop[0][0])] + [("%g,%g" % edge[1])
                for edge in loop]
            li_points += [" ".join(tmp)]
        return ['<polyline points="%s" style="%s" />' % (points, style)
            for points in li_points]

    lines += edge_loops_to_polylines(find_seam_loops(seam),
        styleFormatStr % 0xFF0000)
    lines += edge_loops_to_polylines(find_edge_loops(boundaries),
        styleFormatStr % 0x00FF00)
    lines += edge_loops_to_polylines(find_edge_loops(foldovers),
        styleFormatStr % 0x0000FF)

    lines.append('</svg>')

    return lines


def insert_svg_comments(lineNum, lines, comments):
    """
        Inserts the strings in comments into lines at the lineNum position.
        Inserts the comments in svg comment syntax.
    """
    for comment in comments:
        lines.insert(lineNum, "<!-- %s -->" % comment)
        lineNum += 1


def insert_blank_line(lineNum, lines):
    """ Insert a blank line into the lines list. """
    lines.insert(lineNum, "")

if __name__ == '__main__':

    def usage(afterStr = ""):
        """ Print the usage information and exit the program. """
        print(('Usage: %s path/to/input.obj [-o path/to/output.svg] ' +
            '[-s svg_style]') %
            str(sys.argv[0]), file = sys.stderr)
        print(afterStr, file = sys.stderr)
        sys.exit(-1)

    if len(sys.argv) < 2:
        usage()

    argv = list(sys.argv[1:])

    inpath = argv.pop(0)
    outpath = os.path.splitext(inpath)[0] + '-visualize-seam.svg'
    styleStr = "stroke:#%06x;stroke-width:1;fill:none;"

    try:
        opts, argv = getopt.getopt(argv, "ho:s:")
    except getopt.GetoptError as err:
        usage(err.msg)

    for opt, arg in opts:
        if(opt == "-h"):
            usage()
        elif(opt == "-o"):
            outpath = arg
        elif(opt == "-s"):
            styleStr = arg

    obj = obj_reader.quads_to_triangles(obj_reader.load_obj(inpath))
    linesList = generate_svg_seam(obj, styleFormatStr = styleStr)
    insert_svg_comments(2, linesList,
        [os.path.split(outpath)[1], "Generated by %s" % sys.argv[0],
        "UV Seams for %s" % os.path.split(inpath)[1]])
    insert_blank_line(2, linesList)
    insert_blank_line(6, linesList)
    save_lines(outpath, linesList)
