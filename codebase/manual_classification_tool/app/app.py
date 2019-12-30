from flask import Flask, render_template, redirect, url_for, flash
from wtforms import BooleanField, SubmitField, FormField
from flask_sqlalchemy import SQLAlchemy
from flask import send_from_directory
from flask_wtf import FlaskForm
from flask import Flask
import csv
import os


basedir = os.path.abspath(os.path.dirname(__name__))
app = Flask(__name__)

# Setting up the app with the specific configurations needed to work
# with forms
app.config['SQLALCHEMY_DATABASE_URI'] =\
    'sqlite:///' + os.path.join(basedir,
                                'dbase/email_dataset.sqlite')

app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


db = SQLAlchemy(app)

class Dataset(db.Model):
    """
    The Dataset class inherits from db.Model and implements
    a database that stores an email dataset, which consists of
    a question, an answer (if one exists) and (possible) classes.


    """
    __tablename__ = 'dataset'
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(64), unique=False)
    answer = db.Column(db.String(64), unique=False)
    classes = db.Column(db.String(64), unique=False)


class ClassesForm(FlaskForm):

    """
    This class inherits from FlaskForm and implements a structure
    that contains the form where the class of the question has
    to be selected, it contains manually selected classes.

    """
    class_1 = BooleanField('Class 1')
    class_2 = BooleanField('Class 2')
    class_3 = BooleanField('Class 3')

class EncapsulatedForm(FlaskForm):
    """
    This class inherits from Flaskform and acts as a container
    for the Classesform to allow for easier HTML formatting.
    """
    classes = FormField(ClassesForm)
    submit = SubmitField('Submit')


# This function gets triggered when all emails
# have classes assigned to them in the database
@app.route('/db_complete')
def db_complete():
    return render_template('complete.html')

# This is the main page for the tool
@app.route('/', methods=["GET", "POST"])
def index():
    form = EncapsulatedForm()
    datapoint = Dataset.query.filter(Dataset.classes == "None").first()
    labeled_data = Dataset.query.filter(Dataset.classes != "None").count()
    num_emails = len(Dataset.query.all())
    # this gets triggered if all mails have been classified
    if not datapoint:
        return redirect(url_for('db_complete'))
    email_text = datapoint.question
    retrieved_classes = None

    if form.validate_on_submit():
        retrieved_classes = form.classes.data
        del retrieved_classes['csrf_token']
        cls_list = retrieved_classes.values()

        if any(cls_list):
            datapoint.classes = str(retrieved_classes.values())
            db.session.add(datapoint)
            db.session.commit()
        else:
            # Trigger this alert when the user has not selected any classes
            flash('Gelieve een of meerde klassen te selecteren')
        return redirect(url_for('index'))

    return render_template("index.html", email_text=email_text, form=form,
                           num_labeled=labeled_data,
                           total=num_emails)


# Fixes a warning message about the favicon.ico on firefox
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == "__main__":
    app.run(debug=True)