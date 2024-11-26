push:
	git pull; git commit -am "minor add"; git push

build_docker:
	docker build -t img:full .