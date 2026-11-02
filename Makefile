.PHONY: build run stop test logs
build:
	docker compose build
run:
	docker compose up -d --build
stop:
	docker compose down
test:
	docker compose run --rm backend env -u CAROTID_DICOM_ROOT -u CAROTID_PROJECTS_ROOT python -m pytest -q
	docker compose run --rm --no-deps frontend nginx -t
logs:
	docker compose logs -f
