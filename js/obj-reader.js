/*
Loads an Wavefront OBJ file into a OBJ recordclass.

Written by Yotam Gingold; Zachary Ferguson
*/
"use strict";

// 'extra' is for extra lines
var OBJ = function(v, vt, vn, vc, f, extra, filename){
    return {'v': v, 'vt': vt, 'vn': vn, 'vc': vc, 'f': f, 'extra': extra,
        'filename': filename};
}

var FaceVertex = function(v, vt, vn){
    return {'v': v, 'vt': vt, 'vn': vn};
}

var OBJReader = function OBJReader(){};

OBJReader.parse_obj = function parse_obj(obj_file, filename){
    /* Parses the file object as a OBJ model. */
    if(filename === undefined){
        filename = 'tmp.obj';
    }

    var v = [];
    var vt = [];
    var vn = [];
    var vc = [];
    var f = [];

    var extra_lines = [];

    for(var i = 0; i < obj_file.length; i++){
        var sline = obj_file[i].trim().split(" ").filter(item => item);
        if(sline.length === 0){
            continue;
        }

        if(sline[0] === 'v'){
            v.push(XYZ(...(sline.slice(1, 4).map(parseFloat))));
            if(sline.length > 4){
                vc.push(sline.slice(4).map(parseFloat));
            }
        }
        else if(sline[0] === 'vt'){
            // Could also use UVW coordinates
            vt.push(UV(...(sline.slice(1, 3).map(parseFloat))));
        }
        else if(sline[0] === 'vn'){
            vn.push(XYZ(...(sline.slice(1, 4).map(parseFloat))));
        }
        else if(sline[0] === 'f'){
            // Pad bundle with two extra '//' and then take the first three
            // values in between. This ensures that we always get enough
            // data for a FaceVertex.
            f.push(sline.slice(1).map(bundle => FaceVertex(...(
                    (bundle+'//').split('/').slice(0, 3).map(
                        val => val.length > 0 ? (parseInt(val) - 1):undefined)))));
        }
        else{
            extra_lines.push(obj_file[i].slice())
        }
    }
    return OBJ(v, vt, vn, vc, f, extra_lines, filename);
}


// function load_obj(filename){
//     """ Load a Wavefront OBJ file with the given filename.  """
//
//     print("Loading:", filename)
//
//     with open(filename) as lines:
//         return parse_obj(lines, filename)
// }


OBJReader.quads_to_triangles = function quads_to_triangles(mesh){
    /*
    Convert Quad faces to Triangular ones.
    Inputs:
        mesh - an OBJ object loaded from load_obj()
    Outputs:
        Modifies the mesh.f and returns mesh.
    */
    var newFaces = [];
    for(var i = 0; i < mesh.f.length; i++){
        var face = mesh.f[i];
        if(face.length != 3){
            // assert len(face) == 4
            newFaces.push([face[0], face[1], face[2]]);
            newFaces.push([face[2], face[3], face[0]]);
        }
        else{
            newFaces.push(face);
        }
    }
    mesh.f = newFaces;
    return mesh;
}


// def convert_to_counterclockwise_UVs(mesh):
//     """
//     Converts any clockwise UV triangles to counter-clockwise order.
//     !!! WARNING: This may break find_seam_fast.find_seam() !!!
//     Inputs:
//         mesh - an OBJ object loaded from load_obj(); must be all
//             triangles (use quads_to_triangles())
//     Output:
//         Modifies the mesh.f and returns mesh.
//     """
//     for i, face in enumerate(mesh.f):
//         assert len(face) == 3
//         uvs = [mesh.vt[fv.vt] for fv in face]
//         // Create matrix as specified (http://goo.gl/BDPYIT)
//         mat = numpy.array([[1, uv.u, uv.v] for uv in uvs])
//         if(numpy.linalg.det(mat) < 0): // If order is clockwise
//             mesh.f[i] = (face[1], face[0], face[2]) // Swap order
//     return mesh
