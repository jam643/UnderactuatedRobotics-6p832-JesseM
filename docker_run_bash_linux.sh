#! /bin/bash
echo "Drake Docker container for LINUX"
if [ "$#" != "2" ]; then 
	echo "Please supply two arguments: a Drake release (drake-YYYYMMDD), and a relative path to a directory to mount as /notebooks."
	exit 1
else
	docker pull mit6832/drake-course:$1
	docker run -it --network=host --user=$(id -u) \
	                --env="DISPLAY" \
	                --volume="/etc/group:/etc/group:ro" \
	                --volume="/etc/shadow:/etc/shadow:ro" \
	                --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
	                --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	                --volume="/etc/passwd:/etc/passwd:ro" \
	                --rm \
			        -v "$(pwd)/$2":/notebooks mit6832/drake-course:$1 \
			        /bin/bash -c "cd /notebooks && /bin/bash"
fi
