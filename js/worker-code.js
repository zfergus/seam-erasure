const USING_WORKER = true;

function minimize(file_name, obj_contents) {
//   var file = e.target.files[0];
    var mesh = OBJReader.quads_to_triangles(
        OBJReader.parse_obj(obj_contents.split("\n"), file_name));
    var rows = 100, cols = 100, depth = 3
    var texture = numeric.ones([rows, cols, depth]);
    var out = SeamMinimizer.solve_seam(mesh, texture);
    console.log(numeric.reshape(out, [rows, cols, depth]));
}

self.onmessage = function(msg){
    if(msg.data.id === "input"){
        minimize(msg.data.fname, msg.data.objFile);
    }
    else{
        console.log(msg);
    }
};
