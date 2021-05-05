from os import path


def is_data_exist(directory="large_files"):
    files = [
        "interactions.csv",
        "items.csv",
        "sample_submission.csv",
        "users.csv",
    ]

    for file in files:
        if not path.isfile(path.join(directory, file)):
            return False

    return True
