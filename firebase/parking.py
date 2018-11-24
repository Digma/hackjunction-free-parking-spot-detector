import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# see: https://firebase.google.com/docs/firestore/quickstart


def updateSpacesEspoo(doc_ref, spaces):
    doc_ref.update({
        u'free_spaces': 20-spaces,
        u'used_spaces': spaces
    })