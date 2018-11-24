import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# see: https://firebase.google.com/docs/firestore/quickstart

def updateSpacesEspoo(db, document, spaces):
    doc_ref = db.collection(u'parklots-gael').document(document)
    doc_ref.update({
        u'free_spaces': 20-spaces,
        u'used_spaces': spaces
    })
