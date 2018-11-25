import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# see: https://firebase.google.com/docs/firestore/quickstart

def updateSpaces(db, document, spaces, spaces_taken):
    doc_ref = db.collection(u'parklots').document(document)
    doc_ref.update({
        u'free_spaces': spaces-spaces_taken
    })
