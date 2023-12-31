# Project Title

Form AI

# Project Description

A deep learning model for All Weather horse racing

# Development Environment

1. [Install Python](https://www.python.org/downloads/)
	* Install Pandas (and Numpy) by typing 'pip install pandas' in Windows Command Shell
		* pip install jupyter
	* pip install tqdm
	* pip install matplotlib
		* pip install ipympl
	* pip install tensorflow
	* pip install scikit-learn

2. [Install R](https://cran.ma.imperial.ac.uk/)
	* install.packages("languageserver")
	* install.packages("dplyr")
	* install.packages("mlogit")

3. [Install Visual Studio Code](https://code.visualstudio.com/)
	* [Read Getting Started Guide](https://code.visualstudio.com/docs/?dv=win)
	* Install [Jupyter Notebooks in VSC](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
	* Install R extension for Visual Studio Code

4. [Install Git](https://git-scm.com/download/win) (choose Visual Studio Code as git's default editor, instead of Vim)
	* git config --global user.name "Your Name"
	* git config --global user.email "your.email@example.com"
	* git config --global core.editor "code --wait"

5. [Install MySQL](https://dev.mysql.com/downloads/installer/)
	* pip install mysql-connector (because it failed in MySQL installation step above)
	* Add location of mysql to User Path

6. [Install Smartform database](https://drive.google.com/drive/folders/1BRLT4uH0gtgwt-3MZUrW9raPXsO1-0aM?usp=sharing)
	* CREATE USER 'smartform'@'localhost' IDENTIFIED BY 'smartform';
	* ALTER USER 'smartform'@'localhost' IDENTIFIED WITH mysql_native_password BY 'smartform';
	* CREATE DATABASE smartform;
	* GRANT ALL ON smartform.* TO ‘smartform@’localhost’;
	* mysql -u smartform -p -Dsmartform < historic_data.sql
	* Install SQLAlchemy
		* pip install sqlalchemy
		* pip install pymysql

7. Install PyTorch
	* pip install pytorch	
	* [Install Cuda](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_network) (if you're using a GPU)
		* pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu117

## nbstripout

You can ensure that the output of your Jupyter notebooks is cleared before committing them by using a pre-commit hook with a tool like nbstripout. nbstripout is a tool that can strip the output from Jupyter notebooks.

Here's how you can set it up:

1. Install nbstripout:
	pip install nbstripout 

2. If you want to apply nbstripout to a specific repository, you should navigate to that repository's directory and run:
	nbstripout --install --attributes .gitattributes 

This command will create a .gitattributes file in your repository (or modify it if it already exists) and set up nbstripout as a filter for Jupyter notebooks.
Now, every time you commit a Jupyter notebook, nbstripout will automatically clear its output.
Note: This method doesn't clear the output in your local copies of the notebooks, only in the versions that you commit. If you want to clear the output in your local copies, you should do it manually in Jupyter.
Note: This setup is done on a per-repository basis. If you want to set up nbstripout for all your repositories, you can set it up as a global Git filter.

## What is W&B?

Weights & Biases (W&B) is the AI developer platform, with tools for training models, fine-tuning models, and leveraging foundation models.

Set up W&B in 5 minutes, then quickly iterate on your machine learning pipeline with the confidence that your models and data are tracked and versioned in a reliable system of record.

[Install W&B and start tracking your machine learning experiments in minutes](https://docs.wandb.ai/quickstart)

# Installation

git clone https://github.com/patrickjosephgillen/horseracing.git

# Useful Background

1. [Python](https://docs.python.org/3/tutorial/index.html)
2. [NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html)
3. [Pandas](https://pandas.pydata.org/docs/getting_started/index.html)
4. [Git and GitHub for Beginners - Crash Course](https://www.youtube.com/watch?v=RGOj5yH7evk)
5. historic_data.zip\manual.pdf (SmartForm Database User Manual
6. [PyTorch Crash Course - Getting Started with Deep Learning](https://www.youtube.com/watch?v=OIenNRt2bjg)
7. [MySQL Crash Course | Learn SQL](https://www.youtube.com/watch?v=9ylj9NR0Lcg)
8. [Deep Learning Crash Course for Beginners](https://www.youtube.com/watch?v=VyWAvY2CF9c)
9. Horse Racing 3Q22.html (favourites export) esp.
 	* [All Weather Racing and All Weather Horse Racing Systems](https://betting-sites.me.uk/all-weather-horse-racing-systems/#:~:text=%20Six%20Quick%20Facts%20About%20All%20Weather%20Races,horses%20since%20they%20are%20either%20inexperienced...%20More%20)
	* [Use Machine Learning to Predict Horse Racing](https://towardsdatascience.com/use-machine-learning-to-predict-horse-racing-4f1111fb6ced)
 	* [https://kouohhashi.medium.com/deep-learning-and-horse-race-prediction-1-ed65727f9d27](https://kouohhashi.medium.com/deep-learning-and-horse-race-prediction-1-ed65727f9d27)
 	* [What is backpropagation really doing? | Chapter 3, Deep learning](https://m.youtube.com/watch?v=Ilg3gGewQ5U&t=36s)
	* [Backpropagation calculus | Chapter 4, Deep learning](https://www.youtube.com/watch?v=tIeHLnjs5U8&ab_channel=3Blue1Brown)
