import os

import main
from flask import Flask, flash, request

app = Flask(__name__)

upload_dir = "./"


@app.route("/", methods=['GET'])
def default_route():
    return "<p>DocumentGPT is running!</p>"


@app.route("/get_loaded_files", methods=['GET'])
def get_loaded_files():
    return main.show_all_files()


@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return 'please upload a file.'
    file = request.files['file']
    if file.filename.rsplit('.', 1)[1].lower() not in {'pdf'}:
        return "only pdf files allowed!"

    file_path = os.path.join(app.config['upload_dir'], file.filename)
    file.save(file_path)
    vec_file = main.read_document(file_path)
    return f"file successfully read : {vec_file}"


@app.route('/question', methods=['POST'])
def question():
    request_body = request.json
    if request_body["document_id"] is None:
        return 'document_id is required'

    if request_body["question"] is None:
        return 'document_id is required'

    return main.answer(request_body["document_id"], request_body["question"])


if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.config['upload_dir'] = upload_dir
    app.run()
