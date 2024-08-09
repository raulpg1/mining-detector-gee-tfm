# mining-detector-gee-tfm
Creamos un nuevo enviroment con conda. 
Instalamos cudatoolkit y cudnn para poder usar la gpu en los entrenamientos.
Instalamos todos los paquetes necesarios para ejecutar el proyecto.

1. conda create -n tfm python=3.10 -> 3.9 ->>>>>>>>
2. conda activate tfm
3. conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
4. pip install -r requirements.txt


<<<<<<<<<<< https://www.youtube.com/watch?v=VE5OiQSfPLg <<<<<<<<<<<<<<<<<<<<<<


tensorflow-2.15.0	3.9-3.11	Clang 16.0.0	Bazel 6.1.0	8.9	12.2

https://www.tensorflow.org/install/source


# Update pip
pip install --upgrade pip

# Update Jupyter Notebook or JupyterLab
pip install --upgrade jupyterlab jupyter

# Update ipywidgets
pip install --upgrade ipywidgets


jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter nbextension install --py widgetsnbextension --sys-prefix

jupyter labextension install @jupyter-widgets/jupyterlab-manager
