from __future__ import print_function

import os
import base64
import cStringIO
import time

import numpy
from PIL import Image

from flask import (Flask, request, render_template, url_for, flash, redirect,
    send_file)

from SeamErasure import seam_erasure, obj_reader, util
from SeamErasure.lib import weight_data


app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'tga', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'])


def allowed_file(filename):
    return ('.' in filename and filename.rsplit('.', 1)[1].lower() in
        ALLOWED_EXTENSIONS)


def is_data_file(filename):
    return "data" in request.form or ('.' in filename and
        filename.rsplit('.', 1)[1].lower() == "data")


def upload_file(fileID):
    """ Returns the uploaded file with fileID. None if no file uploaded. """
    if request.method == 'POST':
        if fileID not in request.files:
            return None
        else:
            inFile = request.files[fileID]
            if inFile.filename == '':
                return None
            elif inFile: # and allowed_file(file.filename)
                return inFile
    else:
        return None


@app.route('/')
def index():
    return render_template('min-form.html')


@app.route('/erased', methods=['GET', 'POST'])
def erase():
    if request.method == 'POST':
        try:
            startTime = time.time()
            # Check the uploaded files
            obj_file = upload_file("obj-input")
            if not obj_file or ('.' in obj_file.filename and
                    obj_file.filename.rsplit('.', 1)[1].lower() != "obj"):
                return render_template('min-error.html',
                    error_msg="No OBJ model provided.")

            tex_file = upload_file("tex-input")
            if not tex_file:
                return render_template('min-error.html',
                    error_msg="No texture image provided.")

            mesh = obj_reader.quads_to_triangles(
                obj_reader.parse_obj(obj_file))

            isFloatTexture = isDataFile = False
            if(is_data_file(tex_file.filename)):
                textureData = weight_data.read_tex_from_file(tex_file)[0]
                isFloatTexture, isDataFile = True, True
            else:
                textureData = numpy.array(Image.open(tex_file).transpose(
                    Image.FLIP_TOP_BOTTOM))
                isFloatTexture = not issubclass(textureData.dtype.type,
                    numpy.integer)
                if(not isFloatTexture):
                    textureData = textureData / 255.0
            height, width, depth = (textureData.shape + (1,))[:3]

            sv_methods = {"none": seam_erasure.SeamValueMethod.NONE,
                "texture": seam_erasure.SeamValueMethod.TEXTURE,
                "lerp": seam_erasure.SeamValueMethod.LERP}
            sv_method = sv_methods[request.form["sv"]]

            do_global = "global" in request.form

            out = seam_erasure.erase_seam(mesh, textureData,
                do_global=do_global, sv_method=sv_method,
                display_energy_file=None)

            out = out.reshape((height, width, -1))
            if(out.shape[2] < 2):
                out = numpy.squeeze(out, axis=2)
            if(not isFloatTexture):
                out = util.to_uint8(out)

            base, ext = os.path.splitext(os.path.basename(tex_file.filename))
            out_filename = base + "-erased" + ext
            if isDataFile:
                img_io = cStringIO.StringIO()
                weight_data.write_tex_to_file(img_io, textureData)
                img_io.seek(0)

                return send_file(img_io, as_attachment=True,
                    attachment_filename=out_filename)
            else:
                texture = Image.fromarray(out).transpose(Image.FLIP_TOP_BOTTOM)
                img_io = cStringIO.StringIO()
                texture.save(img_io, format=Image.EXTENSION[ext])
                img_io.seek(0)

                if isFloatTexture:
                    return send_file(img_io, as_attachment=True,
                        attachment_filename=out_filename)

                data_uri = base64.b64encode(img_io.getvalue())
                try:
                    return render_template('min-results.html',
                        min_tex=data_uri, runtime=("%.2f" %
                        (time.time() - startTime)),
                        mime_type=Image.MIME[Image.EXTENSION[ext]])
                except Exception:
                    return send_file(img_io, as_attachment=True,
                        attachment_filename=out_filename)
        except Exception as e:
            return render_template('min-error.html',
                error_msg=("Unable to erase the texture (%s)." % e.message))
    return render_template('min-form.html')

if __name__ == '__main__':
    app.run(debug=True)
