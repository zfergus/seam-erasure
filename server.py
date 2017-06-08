import base64
import cStringIO

import numpy

from flask import Flask, request, render_template
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('min-form.html')


@app.route('/minimized', methods=['GET', 'POST'])
def minimize():
    if request.method == 'POST':
        print(request.files)
        print(request.form)

        from SeamMin import seam_minimizer, obj_reader, util
        obj_file = request.files["obj-input"]
        mesh = obj_reader.quads_to_triangles(obj_reader.parse_obj(obj_file))
        from PIL import Image
        texture = Image.open(request.files["tex-input"]).transpose(
            Image.FLIP_TOP_BOTTOM)
        textureData = numpy.array(texture)
        isFloatTexture = not issubclass(textureData.dtype.type, numpy.integer)
        if(not isFloatTexture):
            textureData = textureData / 255.0
        height, width, depth = (textureData.shape + (1,))[:3]

        sv_methods = {"none": seam_minimizer.SeamValueMethod.NONE,
            "texture": seam_minimizer.SeamValueMethod.TEXTURE,
            "lerp": seam_minimizer.SeamValueMethod.LERP}
        sv_method = sv_methods[request.form["sv"]]

        do_global = "global" in request.form

        out = seam_minimizer.solve_seam(mesh, textureData,
            method=request.form["method"], display_energy_file=None,
            bounds=None, do_global=do_global, sv_method=sv_method)

        out = util.to_uint8(out)
        out = out.reshape((height, width, -1))
        texture = Image.fromarray(out).transpose(Image.FLIP_TOP_BOTTOM)
        buffer = cStringIO.StringIO()
        texture.save(buffer, format="PNG")
        data_uri = base64.b64encode(buffer.getvalue())
        img_tag = '<img src="data:image/png;base64,{0}" style="border-style: solid;border-width:1px">'.format(data_uri)

        return ('<h1>Minimized Texture:</h1>%s' % img_tag)
    return "Error"

if __name__ == '__main__':
    app.run(debug=True)
