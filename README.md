# mining-detector-gee-tfm
Creamos un nuevo enviroment con conda. 
Instalamos cudatoolkit y cudnn para poder usar la gpu en los entrenamientos.
Instalamos todos los paquetes necesarios para ejecutar el proyecto.

1. conda create -n tfm python=3.10
2. conda activate tfm
3. conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
4. pip install -r requirements.txt