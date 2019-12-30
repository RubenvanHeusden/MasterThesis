from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
import csv

# start up the flask application to create the database
basedir = os.path.abspath(os.path.dirname(__name__))
app = Flask(__name__)

# set the name and location of the database, in case of an existing databas, this
# is not necessary
app.config['SQLALCHEMY_DATABASE_URI'] =\
    'sqlite:///' + os.path.join(basedir,
                                'dbase/email_dataset.sqlite')

app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Dataset(db.Model):
    __tablename__ = 'dataset'
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(64), unique=False)
    answer = db.Column(db.String(64), unique=False)
    classes = db.Column(db.String(64), unique=False)


db.drop_all()
db.create_all()
entries = []

with open('dbase/dummy_dataset.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
    next(spamreader, None)
    for row in spamreader:
        entry = Dataset(question=row[0], answer=row[0], classes=row[1])
        entries.append(entry)


db.session.add_all(entries)
db.session.commit()
