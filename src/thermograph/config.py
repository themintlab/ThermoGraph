import os

# Root directory of the thermograph package
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Centralized data paths bundled inside the package
LIBRARY_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, 'library'))
UNARY_TDB_PATH = os.path.abspath(os.path.join(PACKAGE_DIR, 'data', 'unary.tdb'))

# Create the library directory if it doesn't exist and is writable
if not os.path.exists(LIBRARY_DIR):
    try:
        os.makedirs(LIBRARY_DIR, exist_ok=True)
    except OSError:
        pass

