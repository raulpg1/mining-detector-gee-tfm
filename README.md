# mining-detector-gee-tfm
Pasos para la instalación de Tensorflow GPU en Windows 11 a través de WSL2

1. Actualizamos los drives de la GPU (NVIDIA)

2. Creamos un subsistema de Windows para Linux (WSL)
  2.1 wsl --install
  2.2 Setup user and login
  2.3 Update the linux system
    2.3.1 sudo apt-get update
    2.3.2 sudo apt-get upgrade
    2.3.3 sudo reboot

3. Instalamos Anaconda para el manejo de entornos virtuales
  3.1  https://www.anaconda.com/download Linux Python 3.11 64-Bit (x86) Installer (1015.6 MB)
  3.2 Copiamos el archivo al sistema linux
  3.3 Instalamos Anaconda 
    3.3.1 bash Anaconda-latest-Linux-x86_64.sh
    3.3.2 conda config --set auto_activate_base False 
  3.4 Creamos entornos
    3.4.1 conda create -n tfm_tf python=3.11
    3.4.2 conda activate tfm_tf

4. Instalación de CUDA
 4.1  Instalar la versión de CUDA toolkit 12.2
 4.2  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
 4.3  sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
 4.4  wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
 4.5  sudo dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
 4.6  sudo cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
 4.7  sudo apt-get update
 4.8  sudo apt-get -y install cuda

5. Instalación de cuDNN
 5.1 Descargamos la versión de cuDNN v8.9.7
 5.2 Copiamos el archivo descargado en el sistema linux de WSL
 5.3 sudo dpkg -i cudnn-local-repo-$distro-8.9.7.29_1.0-1_amd64.deb
 5.4 sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
 5.5 sudo apt-get update
 5.6 sudo apt-get install libcudnn8=8.9.7.29-1+cudaX.Y
 5.7 sudo apt-get install libcudnn8-dev=8.9.7.29-1+cudaX.Y
 5.8 sudo apt-get install libcudnn8-samples=8.9.7.29-1+cudaX.Y

6. pip install --upgrade pip
7. python3 -m pip install tensorflow[and-cuda]
8. pip install --ignore-installed --upgrade tensorflow==2.15
9. python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
10. conda install -c conda-forge jupyterlab
