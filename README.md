Computational Argumentation 2022 -- Assignment 3
================================================


This directory presents the main structure we expect you to use for handing in the assignment submission. This file describes the expected usage of the main Python file and the usage of the docker image.


## Python file
To make the correction of your assignments easier, please use the provided `main.py` Python file. We will execute the `main.py` when correcting your submission. Also, make sure that all of your code is executed by the `main()` function in the file. You are, of course, free to add more functions and imports if required.


## Evaluation code
To make it easier for you to evaluate your code, we provide a simple evlauation script at `evaluation.py`. If you export your predictions in the appropriate format, this script allows you to retrieve scores on the test set for your predicitons.


## Docker image
To have a unified execution environment for you and us, we also provide a publicly available docker container to execute your submissions in. Please also use this container to run and test your code if possible. Follow the official instructions to set up Docker on your machine [^1]. Below are some useful commands (if you are using Windows, you might need to adapt those commands slightly, but the flags should stay mostly the same).

1. Running the `main.py` file inside the container.
    ```shell
    $ docker run --mount type=bind,src="$(pwd)",dst=/mnt --workdir="/mnt" -it registry.webis.de/code-lib/public-images/upb-ca22:1.0 sh -c 'python main.py'
    ```
2. Installing pip packages inside the container.
    ```shell
    $ docker run --mount type=bind,src="$(pwd)",dst=/mnt --workdir="/mnt" -it registry.webis.de/code-lib/public-images/upb-ca22:1.0 sh -c 'pip install scikit-learn'
    ```
3. Running a jupyter notebook inside the container that is accessible from your browser.
    ```shell
    $ docker run -p 8888:8888 --mount type=bind,src="$(pwd)",dst=/mnt --workdir="/mnt" -it registry.webis.de/code-lib/public-images/upb-ca22:1.0 sh -c 'jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --notebook-dir=/mnt'
    ```
4. Evaluating predictions that are exported to `predictions.json`.
    ```shell
    $ docker run --mount type=bind,src="$(pwd)",dst=/mnt --workdir="/mnt" -it registry.webis.de/code-lib/public-images/upb-ca22:1.0 sh -c 'python evaluation.py --gt_bio_path data/test-bio.csv --pred_bio_path predictions.csv'
    ```



[^1]: https://docs.docker.com/engine/install/