import os
import imghdr
from flask import Flask, render_template, make_response, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import Form
from wtforms import FileField, SubmitField, SelectField, ValidationError, widgets
from wtforms.validators import Required
from subprocess import call

app = Flask(__name__)
app.config['SECRET_KEY'] = 'top secret!'
bootstrap = Bootstrap(app)

lst=[]

def fill_list():
    count = 0
    for file in os.listdir(os.path.join(app.static_folder, "scripts")):
        if file[-2:]=='py':
            lst.append((count,file[:-3]))
            count+=1

class PipelineForm(Form):
    select_op = SelectField('Apply Operation',coerce=int,choices=lst)
    submit = SubmitField('Submit')
    lsit = widgets.ListWidget()


class UploadForm(Form):
    image_file = FileField('Image file',validators=[	])
    submit = SubmitField('Submit')

image = None
pipeline = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global image,pipeline
    # pipeline = []

    uform = UploadForm()
    pform = PipelineForm()

    if pform.validate_on_submit():
        operation = pform.select_op.data
        op = lst[operation][1]
        pipeline.append(op)

        if image != None:
            file_location = os.path.join(app.static_folder, image)
            script_location = os.path.join(app.static_folder, "scripts/"+pipeline[-1]+".py")
            print file_location, script_location
            call(["python",script_location,file_location])
        else:
            pipeline=[]


    elif uform.validate_on_submit():
        if hasattr(uform.image_file.data,'filename'):
            pipeline = []
            for file in os.listdir(app.static_folder+"/temp"):
                os.remove(app.static_folder+"/temp/"+file)
            image = 'temp/' + uform.image_file.data.filename
            uform.image_file.data.save(os.path.join(app.static_folder, image))

    return render_template('index.html', uform=uform, pform=pform, image=image, pipeline = pipeline)

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
    fill_list()
    app.run(debug=True)
