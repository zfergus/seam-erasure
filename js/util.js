/*
Collection of utility functions for wrapping-textures.

Written by Zachary Ferguson
*/

// Record classes for neccessary data
var UV    = function(u, v){ return {'u': u, 'v': v}; };
var Pixel = function(x, y){ return {'x': x, 'y': y}; };
var XY    = function(x, y){ return {'x': x, 'y': y}; };
var XYZ   = function(x, y, z){ return {'x': x, 'y': y, 'z': z}; };

// Quadtratic energy: x.T @ Q @ x + 2 * x.T @ L + C = 0
var QuadEnergy = function(Q, L, C){ return {'Q': Q, 'L': L, 'C': C}; };


// function pairwise(iterable):
//     """ Returns: s -> (s0,s1), (s1,s2), (s2, s3), ... """
//     a, b = itertools.tee(iterable)
//     next(b, None)
//     return zip(a, b)
//
//
// function pairwise_loop(iterable):
//     """
//     Create pair wise list of the iterable given with the last element being the
//     first. Returns: s -> (s0,s1), (s1,s2), (s2, s3), ..., (sN, s0)
//     """
//     return tuple(pairwise(iterable)) + ((iterable[-1], iterable[0]),)

const flatten2D = (arr) => [].concat.apply([], arr);

const product = (...sets) =>
    sets.reduce((acc, set) =>
    flatten2D(acc.map(x => set.map(y => [ ...x, y ]))),
    [[]]);


function rowcol_to_index(row, col, width){
    /* Convert row major coordinates to 1-D index. */
    return width * row + col;
}


function lerp(t, x0, x1){
    /* Linearly interpolate between x0 and x1. */
    return x0 + t * (x1 - x0);
}


// function lerpPair(t, p0, p1){
//     /* Linearly interpolate independent indexed paires */
//     return [lerp(t, p0[0], p1[0]), lerp(t, p0[1], p1[1])];
// }


function lerp_UV(t, uv0, uv1){
    /*
        Linearly interpolate between (u0,v0) and (u1,v1).
        Returns a UV object.
    */
    return UV(lerp(t, uv0.u, uv1.u), lerp(t, uv0.v, uv1.v));
}


function lerp_XY(t, xy0, xy1){
    /*
    Linearly interpolate between (x0,y0) and (x1,y1).
    Returns a XY object.
    */
    return XY(lerp(t, xy0.x, xy1.x), lerp(t, xy0.y, xy1.y));
}


function UV_to_XY(uv, width, height, is_clamped){
    /*
    Converts the given UV to XY coordinates
    uv is defined in terms of GPU UV space.
    */
    // s*width - 0.5; t*height - 0.5
    var xy = XY(x = uv.u * width - 0.5, y = uv.v * height - 0.5);

    if(is_clamped){
        xy = XY(x = Math.min(Math.max(0, xy.x), Math.max(0, width  - 2)),
                y = Math.min(Math.max(0, xy.y), Math.max(0, height - 2)))
    }
    return xy;
}


function UVs_to_XYs(uvEdges, width, height){
    /* Convert a UV edge to XY space in the texture */
    var xys = [];
    for(var i = 0; i < uvEdges.length; i++){
        for(var j = 0; j < uvEdges[i].length; j++){
            xyz.push(UV_to_XY(uvEdges[i][j], width, height));
        }
    }
    return xys;
}


function globalUV_to_local(uv, minX, minY, width, height){
    /*
    Convert from a texture's global UV to local UV.
    Local pixel values defined by the minimum x and y values.
    uv is defined in terms of GPU UV space.
    */
    var xy = UV_to_XY(uv, width, height);
    return UV(xy.x - minX, xy.y - minY);
}


function globalEdge_to_local(uv0, uv1, minI, width, height){
    /*
    Convert a edge from a texture's global UV to local UV.
    Local pixel values defined by the minimum x and y values.
    uv is defined in terms of GPU UV space.
    */
    var minX = minI % width;
    var minY = Math.floor(minI / width);
    return [globalUV_to_local(uv0, minX, minY, width, height),
        globalUV_to_local(uv1, minX, minY, width, height)];
}


function surrounding_pixels(uv, w, h, rtn_format){
    /*
    Determine the surrounding pixels of the given point at (u,v).
    uv is defined in terms of GPU UV space.
    Returns a Tuple of surrounding four Pixel objects.
    Pixels are ordered as: (Lower Left, Lower Right, Upper Left, Upper Right)
    */

    // Convert from GPU UV coordinates to XY coordinates
    var xy = UV_to_XY(uv, w, h, true);
    var x = xy.x, y = xy.y;

    // Convert from XY to Pixel coordinates
    var px = Math.floor(Math.min(Math.max(0, Math.floor(x)), w - 1)); // X in Range(0,w-1)
    var py = Math.floor(Math.min(Math.max(0, Math.floor(y)), h - 1)); // Y in Range(0,h-1)

    var p00 = Pixel(x = px, y = py);

    var px = Math.floor(Math.min(Math.max(0, Math.floor(x) + 1), w - 1)); // X in Range(0,w-1)
    var py = Math.floor(Math.min(Math.max(0, Math.floor(y) + 1), h - 1)); // Y in Range(0,h-1)

    var p11 = Pixel(x = px, y = py);

    // Create tuple of soronding pixels in Pixel Space
    var ps = [p00, Pixel(x = p11.x, y = p00.y), Pixel(x = p00.x, y = p11.y), p11]

    // If requested, convert from Pixel space to 1D index space
    if(rtn_format === "index"){
        return ps.map(p => rowcol_to_index(p.y, p.x, w));
    }
    if(rtn_format === "array"){
        return ps.map(p => [p.x, p.y]);
    }
    return ps;
}


function range_min_max(a, b){
    /* Creates a range from the min value to the max value. */
    return numeric.range(Math.floor(Math.min(a, b)), Math.floor(Math.max(a, b)));
}


// function print_dots(time_delta = 1.0){
//     /*
//     Print out a dot every time_delta seconds.
//     Loop after three dots.
//     */
//     dot_count = 0
//     while True{
//         dot_count = (dot_count % 3) + 1
//         print(("." * dot_count) + (" " * 3), end = "\r")
//         sys.stdout.flush()
//         time.sleep(time_delta)
//     }
// }


function round(value, decimals) {
    return Number(value.toFixed(decimals));
}


function print_progress(percent){
    /*
    Prints a dot followed by the given percentage.
    Given value should be a decimal in range [0, 1].
    */
    var out_str = "\r" + round(percent * 100, 2) + "%";
    log_output(out_str);
}

function log_output(str, useConsole){
    if(useConsole){
        console.log(str);
    }
    else if(typeof(USING_WORKER) !== "undefined"){
        self.postMessage({'id': 'output', 'out_str': str});
        // console.log(str);
    }
    else{
        var outputEl = document.getElementById("output");
        if(outputEl !== null){
            if(outputEl.innerHTML === ""){
                outputEl.innerHTML = "<h3>Output:</h3>"
            }
            outputEl.innerHTML += str.replace("\n", "<br>") + "<br>";
        }
        console.log(str);
    }
}


// function print_clear_line(line_length = 80){
//     /* Clear the current line with 80 spaces followed by a carage return. */
//     print("\r" + (" " * line_length) + "\r", end = "")
// }


// !!! These functions are not useful !!!
// function texUV_to_gpuUV(uv, width, height){
//     /* Convert from the Texture UV space to GPU/OpenGL UV space. */
//     u = uv.u - (uv.u / float(width)) + 0.5 / width
//     v = uv.v - (uv.v / float(height)) + 0.5 / height
//     return UV(u = u, v = v)
//
//
// function texSeam_to_gpuUV(seam, width, height){
//     /* Convert a texture seam to GPU/OpenGL UV space. */
//     gpu_seam = list()
//     for edgePair in seam{
//         gpu_edgePair = list()
//         for edge in edgePair{
//             gpu_edgePair.append(
//                 [texUV_to_gpuUV(uv, width, height) for uv in edge])
//         gpu_seam.append(gpu_edgePair)
//     return gpu_seam


// function verts_equal(v0, v1, epsilon = 1e-8){
//     /*
//     Test if two given vertices are equal within a certain epsilon.
//     WARNING:
//         This is slower than ==, but it allows for a tolerance level of
//         equality.
//     */
//     assert epsilon >= 0.0
//
//     if len(v0) != len(v1){
//         return False
//
//     for a, b in zip(v0, v1){
//         if(abs(a - b) > epsilon){
//             return False
//     return True
// }

// function normalize_array(arr){
//     /* Normalize the given array to be in range [0,1]. */
//     minVal = numpy.amin(arr)
//     maxVal = numpy.amax(arr)
//     return (arr - minVal) / float(maxVal - minVal)
// }


function is_counterclockwise(v0, v1, v2){
    /*
    Is the triangle defined by the given vertices in counter-clockwise order?
    Input:
        v0, v1, v2 - 2D coordinates for the vertices of the triangle
    Output:
        Returns True if the triangle is counter-clockwise order.
    */
    var mat = [[1, v0[0], v0[1]],
               [1, v1[0], v1[1]],
               [1, v2[0], v2[1]]];
    return numeric.det(mat) > 0;
}


// Convert back to image format
function to_uint8(data){ //, normalize = False){
    /* Convert the data in a floating-point vector to unsigned bytes. */
    // Normilize the solved values.
    // if(normalize){
    //     data = normalize_array(data)
    // }

    // TODO: Properly cast each element as a 8-bit unsigned integer.
    data = numeric.saturate(data);
    data = numeric.round(numeric.mul(data, 255)); //.astype('uint8');
    return data;
}


// function save_ijvs(A, fname){
//     /* Save a sparse matrix as a list of ijv pairings. */
//     A = A.tocoo()
//     height, width = A.shape
//     M = numpy.empty((A.row.shape[0], 3))
//     M[:, 0] = A.row
//     M[:, 1] = A.col
//     M[:, 2] = A.data
//     lines = ["%d %d %.17f\n" % (ijv[0], ijv[1], ijv[2]) for ijv in M]
//     with open(fname, "w") as f{
//         f.write("%d %d\n" % (height, width))
//         for line in lines{
//             f.write(line)
// }
//
// function save_dense(A, fname){
//     /* Saves an array as a text file, one line per row. */
//     m, n = A.shape
//     with open(fname, "w") as f{
//         for row in A{
//             for val in row{
//                 f.write("%.17f " % val)
//             f.write("\n")
// }
