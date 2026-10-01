# Convenience targets. On Windows use the commands inside directly (or WSL).
PY ?= python
DATA = data/extention of Z-Alizadeh sani dataset.xlsx

.PHONY: install run train test e2e glb
install:           ## create nothing, just install dependencies into the active environment
	$(PY) -m pip install -r requirements.txt
run:               ## start the app on http://localhost:8000
	$(PY) -m uvicorn cardio.serve:app --port 8000
train:             ## nested cross-validation + final models (about 5 minutes on one core)
	$(PY) -m cardio.train --data "$(DATA)"
test:              ## fast Python tests (about 1 minute)
	$(PY) -m pytest -q tests
e2e:               ## headless-browser test; start `make run` in another terminal first
	cd tests/e2e && npm install && npm test
glb:               ## shrink the skeleton in thorax.glb (needs trimesh, fast-simplification)
	$(PY) scripts/optimize_glb.py
