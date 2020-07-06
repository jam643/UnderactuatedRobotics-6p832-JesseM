These folders contain my solutions to Russ Tedrake's Underactuated Robotics class 6.832 at MIT. The Jupyter notebook homeworks assignements are run via shared volume in docker container containing pyDrake. To run on linux machine with docker installed:

- Run `./docker_run_notebook.sh drake-20190508 .` in the project directory to start the specified docker Drake container, mounting the current directory as the shared volume.
- Enter `http://127.0.0.1:8080/` in a browser
- Password is mit6832
