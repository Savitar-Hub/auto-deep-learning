upload:
		rm -rf dist
		python3 setup.py sdist bdist_wheel 
		twine check dist/*
		twine upload dist/*

test:
		python3 -m pytest tests