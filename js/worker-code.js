const USING_WORKER = true;

function minimize(obj_contents, texture) {
    var mesh = OBJReader.quads_to_triangles(
        OBJReader.parse_obj(obj_contents.split("\n"), ""));
    var dims = numeric.dim(texture).concat(1);
    var nRows = dims[0], nCols = dims[1], depth = dim[2];

    var out = SeamMinimizer.solve_seam(mesh, texture);
    console.log(numeric.reshape(out, [nRows, nCols, depth]));
}

self.onmessage = function(msg){
    if(msg.data.id === "input"){
        minimize(msg.data.objFile, JSON.parse(msg.data.texture));
    }
    else{
        console.log(msg);
    }
};
