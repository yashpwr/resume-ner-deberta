.PHONY: setup fetch merge train eval predict serve onnx clean

setup:
	pip install -r requirements.txt

fetch:
	python scripts/fetch_data.py

merge:
	python scripts/merge_datasets.py

train:
	python -m src.train --config configs/train.yaml

eval:
	python -m src.eval

predict:
	python -m src.predict --text "John worked at Google as a Software Engineer"

serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

onnx:
	python scripts/export_onnx.py

clean:
	rm -rf data/ artifacts/ __pycache__/ .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

test:
	python -m pytest tests/ -v

all: setup fetch merge train eval
