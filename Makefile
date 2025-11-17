.PHONY: serve frontend train clean

serve:
	cd src/model-backend && uvicorn server:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd src/frontend && dotnet run

train:
	cd model && jupyter nbconvert --to notebook --execute model.ipynb --output model_trained.ipynb

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name "sp500_data_*.csv" -not -name "sp500.csv" -delete 2>/dev/null; true
