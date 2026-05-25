.PHONY: train test lint docker-build docker-run k8s-deploy

train:
	python train.py

test:
	pytest -v

lint:
	ruff check .

docker-build:
	docker build -t rakibulinux/diabetes-api .

docker-run:
	docker run -p 8000:8000 rakibulinux/diabetes-api

k8s-deploy:
	kubectl apply -f k8s-deploy.yml
