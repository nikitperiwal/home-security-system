import pickle


def save_faces(face_dict):
    with open("registered_faces.pickle", "wb") as f:
        pickle.dump(face_dict, f)


def load_faces():
    try:
        with open("registered_faces.pickle", "rb") as file:
            unpickler = pickle.Unpickler(file)
            face_dict = unpickler.load()
            return face_dict
    except FileNotFoundError:
        print("No faces registered")
        return {}
