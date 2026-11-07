.PHONY: setup data train report app clean

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

data:
	.venv/bin/python scripts/fetch_data.py
	.venv/bin/python src/data_prep.py
	.venv/bin/python src/features.py

train:
	.venv/bin/python src/train.py

report:
	.venv/bin/python src/report_generator.py

app:
	.venv/bin/streamlit run app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
