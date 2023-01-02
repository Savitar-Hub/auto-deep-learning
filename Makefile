# Upload the package into Pip
upload:
		rm -rf dist
		python3 setup.py sdist bdist_wheel
		twine check dist/*
		twine upload dist/*
# TODO: Upload with also a tag of the GH repo

# Make the testing of the package
test:
		python3 -m pytest tests

# Make count of lines of the program: apt install cloc
count:
		cloc . --exclude-dir=venv --exclude-ext=csv
