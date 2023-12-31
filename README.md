# Project Title

Form AI

# Project Description

A deep learning model for All Weather horse racing

# Development Environment

	[Install Python](https://www.python.org/downloads/)
		Install Pandas (and Numpy) by typing 'pip install pandas' in Windows Command Shell
		pip install jupyter
		pip install tqdm
		pip install matplotlib
		  pip install ipympl
		pip install tensorflow
		pip install scikit-learn
	[Install R](https://cran.ma.imperial.ac.uk/)
		install.packages("languageserver")
		install.packages("dplyr")
		install.packages("mlogit")
	[Install Visual Studio Code](https://code.visualstudio.com/)
		[Read Getting Started Guide](https://code.visualstudio.com/docs/?dv=win)
		Install [Jupyter Notebooks in VSC](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
		Install R extension for Visual Studio Code
	[Install Git](https://git-scm.com/download/win) (choose Visual Studio Code as git's default editor, instead of Vim)
		git config --global user.name "Your Name"
		git config --global user.email "your.email@example.com"
		git config --global core.editor "code --wait"
	[Install MySQL](https://dev.mysql.com/downloads/installer/)
		pip install mysql-connector (because it failed in MySQL installation step above)
		Add location of mysql to User Path
	Install Smartform database
		CREATE USER 'smartform'@'localhost' IDENTIFIED BY 'smartform';
		ALTER USER 'smartform'@'localhost' IDENTIFIED WITH mysql_native_password BY 'smartform';
		CREATE DATABASE smartform;
		GRANT ALL ON smartform.* TO ‘smartform@’localhost’;
		mysql -u smartform -p -Dsmartform < historic_data.sql
	Install SQLAlchemy
		pip install sqlalchemy
		pip install pymysql
	Install PyTorch
	  pip install pytorch	
    [Install Cuda](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_network) (if you have a GPU)
		pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu117

# Installation

git clone https://github.com/patrickjosephgillen/horseracing.git

# Useful Background

