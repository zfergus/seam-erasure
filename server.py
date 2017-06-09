import base64
import cStringIO
import time

import numpy
from PIL import Image

from flask import Flask, request, render_template, url_for, flash, redirect
from werkzeug.utils import secure_filename

from SeamMin import seam_minimizer, obj_reader, util
from SeamMin.lib import weight_data


app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'tga', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'])


def allowed_file(filename):
    return ('.' in filename and filename.rsplit('.', 1)[1].lower() in
        ALLOWED_EXTENSIONS)


def is_data_file(filename):
    return "data" in request.form or ('.' in filename and
        filename.rsplit('.', 1)[1].lower() == "data")


def upload_file(fileID, flashMSG):
    """ Returns the uploaded file with fileID. None if no file uploaded. """
    if request.method == 'POST':
        if fileID not in request.files:
            # flash(flashMSG)
            return None
        else:
            inFile = request.files[fileID]
            if inFile.filename == '':
                # flash(flashMSG)
                return None
            elif inFile: # and allowed_file(file.filename)
                return inFile
    else:
        return None


@app.route('/')
def index():
    return render_template('min-form.html')


@app.route('/minimized', methods=['GET', 'POST'])
def minimize():
    if request.method == 'POST':
        startTime = time.time()
        # Check the uploaded files
        obj_file = upload_file("obj-input", "No OBJ model provided.")
        if not obj_file or ('.' in obj_file.filename and
                obj_file.filename.rsplit('.', 1)[1].lower() != "obj"):
            return redirect(request.url)

        tex_file = upload_file("tex-input", "No texture image provided.")
        if not tex_file:
            return render_template('min-form.html')

        mesh = obj_reader.quads_to_triangles(obj_reader.parse_obj(obj_file))

        isFloatTexture = isDataFile = False
        if(is_data_file(tex_file.filename)):
            textureData = weight_data.read_tex_from_path(fpath)[0]
            isFloatTexture, isDataFile = True, True
        else:
            textureData = numpy.array(Image.open(tex_file).transpose(
                Image.FLIP_TOP_BOTTOM))
            isFloatTexture = not issubclass(textureData.dtype.type,
                numpy.integer)
            if(not isFloatTexture):
                textureData = textureData / 255.0
        height, width, depth = (textureData.shape + (1,))[:3]

        sv_methods = {"none": seam_minimizer.SeamValueMethod.NONE,
            "texture": seam_minimizer.SeamValueMethod.TEXTURE,
            "lerp": seam_minimizer.SeamValueMethod.LERP}
        sv_method = sv_methods[request.form["sv"]]

        do_global = "global" in request.form

        out = seam_minimizer.solve_seam(mesh, textureData,
            display_energy_file=None, do_global=do_global, sv_method=sv_method)

        out = out.reshape((height, width, -1))
        if(out.shape[2] < 2):
            out = numpy.squeeze(out, axis=2)
        if(not isFloatTexture):
            out = util.to_uint8(out)
        if(isDataFile or isFloatTexture):
            # flash("Data download not implemented.")
            return render_template('min-form.html')
        else:
            texture = Image.fromarray(out).transpose(Image.FLIP_TOP_BOTTOM)
            buffer = cStringIO.StringIO()
            texture.save(buffer, format="PNG")
            data_uri = base64.b64encode(buffer.getvalue())
            return render_template('min-results.html', min_tex=data_uri,
                runtime=("%.2f" % (time.time() - startTime)))
    return render_template('min-form.html')

if __name__ == '__main__':
    app.run(debug=True)
