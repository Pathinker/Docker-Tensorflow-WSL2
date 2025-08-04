# Docker Tensorflow WSL2

[![EN](https://img.shields.io/badge/EN-English-blue?style=for-the-badge)](https://github.com/Pathinker/Docker-Tensorflow-WSL2/blob/main/README.MD)
[![ES](https://img.shields.io/badge/ES-Español-red?style=for-the-badge)](https://github.com/Pathinker/Docker-Tensorflow-WSL2/blob/main/README.ES.MD)

<p align="justify">    
Tensorflow tiene múltiples desafíos para habilitar el uso de la GPU en entrenamiento o inferencia de los modelos, usualmente usar una gráfica o unidad tensorial reduce por 10 veces o más el tiempo de ejecución
</p>

<p align="justify">
Desde la versión <b>2.11.0</b> el soporte nativo de <b>GPU en sistemas de Windows fue removido</b> y con ello son sugeridos múltiples métodos para retornar la compatibilidad como lo son crear un entorno virtual en conda con las versiones específicas de Python, no obstante, no es posible instalar las nuevas características después de la versión 2.11.0 lanzada en 2022. Otras soluciones podrían ser usar <b>Google Colab</b> que ya trae configurado todo lo necesario para ejecutar un entorno con GPU o hacer uso de <b>subsistema de Windows para Linux (WSL2)</b> que es incluido en Windows para correr cualquier distribución de Linux.
</p>

<p align="justify">
En este caso será explicado como instalar y manejar apropiadamente WSL2 para habilitar el soporte de GPU y compatibilidad con TensorFlow u otros frameworks o tecnologías, encontraras una guía paso por paso para construir un entorno virtual con Docker ejecutándose en WSL2 o en Visual Studio Code en un Dev Container, extensión desarrollada por Microsoft.
</p>

## Summary

- ¿Cómo instalarlo?
  - $${\color{orange}1.- \space Docker \space Desktop}$$
  - $${\color{orange}2.- \space WSL2}$$
  - $${\color{orange}3.- \space Nvidia \space Container \space Toolkit}$$
  - $${\color{orange}4.- \space Verificar \space Instalación \space Docker \space}$$
- Docker Explicación
  - $${\color{royalblue}Animación}$$
  - $${\color{orange}Docker \space Proceso}$$
    - $${\color{royalblue}Animación}$$
    - $${\color{lightskyblue}Dockerfile}$$
    - $${\color{lightskyblue}Docker \space Imagen}$$
    - $${\color{lightskyblue}Docker \space Contenedor}$$
      - $${\color{royalblue}Animación}$$
      - $${\color{blue}Parametros}$$
      - $${\color{royalblue}Animación}$$
- Ejecutar Proyecto (Escoge Entorno)
  - $${\color{orange}Ubuntu \space Terminal}$$
  - $${\color{orange}Visual \space Studio \space Code \space (DevContainer)}$$
- Video

# ¿Cómo instalarlo?

<p align="justify">
En orden para ejecutar y replicar los resultados del proyecto fue virtualizado mediante Docker siendo necesario solo instalar las herramientas para manejar el entorno virtual, además un dev container fue realizado en visual studio para requerir menos comandos y experimentar en un IDE más sencillo.
</p>

## $${\color{orange}1.- \space Docker \space Desktop}$$

<p align="justify">
Descarga y procede a instalar <a href = "https://www.docker.com"> Docker desktop </a> desde su página oficial, una vez concluido habilita la conexión con WSL2, actualmente en “<b>General</b>" y haz click sobre "<b>Use the WSL2 based engine</b>" o similar.
</p>

## $${\color{orange}2.- \space WSL2}$$

<p align="justify">

WSL2 es el subsistema de Windows para Linux, esto permite integrar y usar múltiple kernel de diferentes distribuciones de Linux sin la necesidad de una máquina virtual y desde Windows nativo, para abrirlo es necesario realizar lo siguiente:
</p>

- Habilita "**Plataforma de Máquina Virtual**" y "**Subsistema de Windows para Linux**" en características de Windows, ubicado en "**Activar o Desactivar las Características de Windows**"
- Abre el PowerShell  de Windows como administrador y escribe "**wsl --install**"
- Logeate en la distribución de Ubuntu como un nuevo usuario y escribe "**sudo apt update && sudo apt upgrade -y**" para descargar todas las actualizaciones.

Más información en sitio oficial de <a href = "https://learn.microsoft.com/en-us/windows/wsl/install"> Microsoft </a>.

## $${\color{orange}3.- \space Nvidia \space Container \space Toolkit}$$

<p align="justify">
Nvidia Container Toolkit permite a los contenedores de Docker usar todas las unidades graficas.
</p>

<p align="justify">
Copia los siguientes comandos para descargar e instalar los archivos al final reinicia el motor de Docker.
</p>

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo apt-get update
```

```
sudo apt-get install -y nvidia-container-toolkit
```

Más información en sitio oficial de <a href = "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"> Nvidia </a>.

## $${\color{orange}4.- \space Verificar \space Instalación \space Docker}$$

- Dirígete a la distribución de Ubuntu en WSL2 y digita "**```docker --version```**" si todo esta correcto deberías de ver la versión actual de Docker en tu sistema.
- Ejecuta la siguiente imagen de Docker "**```docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi```**" si logras visualizar tu gráfica, drivers y versión de Cuda todo ha sido instalado correctamente.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c1109a57-f7b6-4b87-98f8-50d7c4c668fe" alt="image">
</p>

# Docker Explanation

<p align="justify">
Docker es una herramienta que permite aislar entornos virtuales al igual que Conda pero brindando una mayor versatilidad y compatibilidad entre computadores, esto significa que cada ordenador ejecutara de la misma manera lo que fue configurado usando Docker, brindando una mejor calidad de vida al software y en este caso particular te permitirá disponer de múltiples personas trabajando en el mismo proyecto sin ningún problema de compatibilidad de versiones al incorporar un nuevo dispositivo al trabajo, además de brindar reproducibilidad al software.
</p>

>[!IMPORTANT]
> En el ámbito científico es significativo la replicación de los resultados experimentales.

<p align="justify">
Docker puede ser llevado más lejos con la extensión <b>Dev Container</b> la cual habilita exportar Docker al IDE de Visual Studio Code e incluso incorporar ajustes personalizados o extensiones personales o de calidad de servicio, su beneficio respecto máquinas virtuales es la reutilización del kernel del sistema operativo, mientras se mantiene aislado cada instancia de Docker, esto permite incrementar la optimización de recursos y con otros cualidades de <b>Docker Compose</b> y <b>Docker Swarm</b> brinda a software productivo la habilidad de escalar y adaptarse a los requerimientos y demanda.
</p>

https://github.com/user-attachments/assets/fc03ff05-4119-4dcc-9d09-b0900088cdf2

## $${\color{orange}Docker \space Proceso}$$

<p align="justify">
Docker toma diferentes etapas para concretar y crear entornos virtuales siendo, Dockerfile, Imagen y Contenedor.
</p>

https://github.com/user-attachments/assets/9f1a47e9-2737-4652-a91a-f8758c621232

### $${\color{lightskyblue}Dockerfile}$$

<p align="justify">
  Detalla toda las instrucciones de configuración con las siguientes palabras claves:
</p>

  - **FROM:** Escrito en el inicio hereda todas las configuraciones de una imagen previa de Docker alojada en Docker Hub, esto permite ahorrar tiempo y reduce el tamaño y complejidad del archivo de Docker, es posible realizar multiples FROM mediante un proceso llamado [Multistage](https://github.com/Pathinker/Docker-Tensorflow-WSL2/tree/main/.docker).
  - **WORKDIR:** Especifica el directorio de trabajo donde serán copiados y redirigidos todos los datos.
  - **COPY:** Copia todos los archivos entre el host y el entorno virtual.
  - **RUN:** Permite ejecutar comandos de terminal como lo son sudo update, upgrade o pip install.
  - **CMD:** Brinda de una rutina predeterminada al ejecutarse un entorno.
  - **EXPOUSE:** Habilita un puerto para su uso, por ejemplo 8888 para Jupyter Lab.

### $${\color{lightskyblue}Docker \space Imagen}$$

<p align="justify">
Contiene todas y cada una de las configuraciones del Dockerfile listo para ejecutar el entorno virtual de la imagen, requiere tener un nombre especifico con los siguientes elementos: "usuario/nombre-proyecto:tag":
</p>

  - **usuario:** Opcional, indica el usuario propietario de la imagen, es necesario cuando una imagen es subida a la nube en Docker Hub para ser descargada por más usuarios.
  - **nombre-proyecto:** Encapsula el nombre del repositorio.
  - **tag:** Especifica el propósito o información relacionada a características.

La imagen es obtenida con el siguiente comando:

```
docker build -t {docker-nombre-imagen} {dockerfile-directorio}
```

>[!TIP]
> La imagen puede no usar cache de construcciones previas con --no-cache y compatibilidad con múltiples plataformas con --platform-linux/amd64,linux/arm64

### $${\color{lightskyblue}Docker \space Contenedor}$$

<p align="justify">
Despliega una instancia de una imagen con un entorno virtual corriendo características especificadas, varios contenedores pueden ser creados desde una misma imagen y administrados con Docker Compose, Docker Swarm y Kubernetes siendo capaz de segmentar un software en múltiples contenedores individuales acorde la arquitectura de software de microservicios.
</p>

https://github.com/user-attachments/assets/a3312a95-f922-4f9d-87ea-147b63ca3ddf

Para ejecutar un contenedor es necesario una serie de parámetros.

```
docker run -it --rm --gpus all -v {dirección-absoluta-computador-host:WORKDIR} --name {nombre-deseado-al-contenedor} {nombre-imagen-docker}
```

- **it:** Permite una terminal interactiva, será capaz de mostrar retroalimentación y registros de la terminal.
- **rm:** Remueve un contenedor al cerrarlo.
- **gpus all:** Brinda acceso a todas las unidades graficas del dispositivo usando Nvidia Container Toolkit.
- **v {absolute-direction-host-machine:WORKDIR}:** Crea un volumen con un directorio especificado del ordenador host, esto hace posible almacenar, actualizar y crear datos mientras el contenedor es ejecutado.
- **name:** Nombra un contenedor.

>[!TIP]
> Para ejecutar desde la terminal y evitar el entrypoint o la instrucción CMD del Dockerfile y la imagen de Docker, incorpora "/bin/bash" al final del comando "docker run -it ... /bin/bash".

https://github.com/user-attachments/assets/320895dc-a319-40eb-9ebc-47f5c833d2bb

# Ejecutar Proyecto (Escoge Entorno)

<p align="justify">
Ahora tienes dos opciones para ejecutar el código, una siendo correr directamente en la terminal nativa de Ubuntu que ya se encuentra instalada o agregar una extension a Visual Studio Code para obtener un entorno de desarrollo más cómodo.
</p>

## $${\color{orange}Ubuntu \space Terminal}$$

<p align="justify">
Ejecuta el siguiente comando y espera alrededor de 30 minutos para descargar completamente la imagen de Docker.
</p>

```
docker run -it --rm --gpus all --name tensor pathinker/docker-tensorflow-wsl2:2.17.0
```

<p align="justify">
Ahora necesitas escribir python3 y la ruta completa de los archivos a ejecutar y los otros comandos para editar, visualizar archivos, de igual manera todo el código estará totalmente operativo.
</p>

> [!CAUTION]
> Nota: Todos los cambios una vez el programa este ejecutándose serán perdidos por la ausencia de un volumen que comparte la información entre el entorno virtual y el dispositivo host.

## $${\color{orange}Visual \space Studio \space Code \space (DevContainer)}$$

<p align="justify">
Para ejecutarlo en Visual Studio Code necesitaras buscar una extensión y escribir "<b><a href = "https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers">Dev Containers</a></b>" de Microsoft, espera hasta que se encuentre totalmente funcional y después realiza el siguiente atajo "<b>Ctrl + Shift + P</b>", escribe y haz click en "<b>>Dev Containers: Rebuild and Reopen in Container</b>". Después necesitaras esperar alrededor del mismo tiempo que en la terminal de Ubuntu.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e5c8dd9f-792c-4b8e-9a95-9fd4aa7d2883" alt="image">
</p>

<p align="justify">
Usando los dev container en Visual Studio Code brinda de unas pocas extensiones para habilitar el debug de python, botones de ejecución rápidos, acceso a las claves SHSH para clonar y modificar el repositorio, terminal zsh y la persistencia ante cambios y no abrir la terminal de WSL2 o Ubuntu.
</p>

> [!IMPORTANT]
> El engine de Docker debe estar ejecutándose siempre si deseas reabrir el proyecto en ambos escenarios.

# Video

[![Docket-Tensorflow-WSL2-Video](https://github.com/user-attachments/assets/9d67e5bd-0c52-49bc-8af9-990ebc75feeb)](https://youtu.be/acDmVYKnIVc)
