# Customer Support Copilot — top-level entry points

.PHONY: install chunks train-biencoder train-fusion train-dora train-router train eval-baseline eval-full compare test demo api clean

PY := python3

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

chunks:
	jupyter nbconvert --to notebook --execute notebook/main.ipynb --output main_executed.ipynb

train-biencoder:
	$(PY) -m training.train --stage biencoder --config config/full.yaml

train-fusion:
	$(PY) -m training.train --stage fusion    --config config/full.yaml

train-dora:
	$(PY) -m training.train --stage dora      --config config/full.yaml

train-router:
	$(PY) -m training.train --stage router    --config config/full.yaml

train: train-biencoder train-fusion train-dora train-router

eval-baseline:
	$(PY) -m training.evaluation --config config/baseline.yaml

eval-full:
	$(PY) -m training.evaluation --config config/full.yaml

compare:
	$(PY) -m training.evaluation --compare

eval: eval-baseline eval-full compare

test:
	$(PY) -m pytest tests/ -v

demo:
	$(PY) -m demo.cli --config config/full.yaml

api:
	uvicorn demo.app:app --host 0.0.0.0 --port 8000

clean:
	rm -rf experiments/results/*.json
	rm -rf data/processed/faiss_index*.bin
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
