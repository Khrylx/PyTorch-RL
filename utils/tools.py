from os import path


def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))
