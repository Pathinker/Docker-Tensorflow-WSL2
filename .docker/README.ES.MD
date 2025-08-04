# Docker Multistage

[![EN](https://img.shields.io/badge/EN-English-blue?style=for-the-badge)](https://github.com/Pathinker/Docker-Tensorflow-WSL2/blob/main/.docker/README.MD)
[![ES](https://img.shields.io/badge/ES-Español-red?style=for-the-badge)](https://github.com/Pathinker/Docker-Tensorflow-WSL2/blob/main/.docker/README.ES.MD)

<p align="justify">
Docker brinda la posibilidad de cargar múltiples configuraciones de imágenes, esto es realizado para ahorrar tiempo y líneas en configuraciones específicas, para hacer esto posible cada FROM que es llamado es una etapa. Donde la imagen es construida y de manera nativa brinda un numero natural a cada etapa empezando con el 0 pero es posible asignar un nombre.
</p>

<p align="justify">
Después de eso solamente es necesario copiar la configuración en específica del directorio en la última etapa, en este ejemplo es realizado un multistage para brindar a configuraciones de python al sistema operativo Ubuntu 22.04 para una instalación simple de Cuda y cuDNN, en lugar de Debian 12 que la imagen original tiene.
</p>

>[!Warning]
> Este proceso solamente es realizado cunado la imagen de tensorflow no tiene las versiones correctas de Cuda y cuDNN que son necesarias para detectar y brindar soporte a la GPU.

https://github.com/user-attachments/assets/944c2d9e-a2c0-4624-9c9d-d9f1dfa0f612