import pickle
from multiprocessing import Manager


def verify_register_person(name, face_images, registered):
    """"
    Verifies the Register_Person parameter.
    """
    if not isinstance(name, str):
        raise ValueError("name should have d-type: str")
    if len(name) == 0:
        raise ValueError("name cannot be empty")
    if name in registered:
        raise ValueError("Name already exists in Secure Faces List\nPlease enter another Name")
    if face_images > 3:
        print("Only first 3 images for the person would be registered")


def verify_delete_person(name, registered):
    """"
    Verifies the Delete_Person parameter.
    """
    if not isinstance(name, str):
        raise ValueError("name should have d-type: str")
    if len(name) == 0:
        raise ValueError("name cannot be empty")
    if name not in registered:
        raise ValueError("Person with that name doesn't exists")


def verify_rename_person(old_name, new_name, registered):
    """"
    Verifies the Rename_Person parameter.
    """
    if not isinstance(old_name, str):
        raise ValueError("old_name should have d-type: str")
    if not isinstance(new_name, str):
        raise ValueError("new_name should have d-type: str")
    if len(new_name) == 0:
        raise ValueError("new_name cannot be empty")
    if old_name not in registered:
        raise ValueError(f"Person with name: {old_name} doesn't exists")
    if new_name in registered:
        raise ValueError(f"Person with name: {new_name} already exists")


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
        print("No registered faces found")
        manager = Manager()
        return manager.dict()
