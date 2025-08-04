# Docker Tensorflow WSL2

<p align="justify">    
Tensorflow has multiple issues enabling GPU support training or inference from their models, usually using a graphic or tensorial unit reduces by 10 times or more the running time.
</p>

<p align="justify">
Since their version <b>2.11.0</b> the native <b>GPU support for windows systems was removed</b> and suggested multiply methods to gain the compatibility again such being in a virtual environment with Conda and the specifics versions of python, nevertheless it is not possible to install the new features coming towards the 2.11.0 released in 2022. Other solutions could be using <b>Google Colab</b> which has already everything configured to run with a GPU environment or employ the <b>windows subsystem for Linux (WSL2)</b> that is included with windows to run Linux distributions.
</p>

<p align="justify">
In this case is explained how to properly install and manage the WSL2 to enable the GPU support and compatibility with TensorFlow or other frameworks or technologies, you will find a guide step by step to build a virtual environment with Docker running in WSL2 or Visual Studio Code with the Dev Container extension developed by Microsoft.
</p>

## Summary

- How to Install
  - $${\color{orange}1.- \space Docker \space Desktop}$$
  - $${\color{orange}2.- \space WSL2}$$
  - $${\color{orange}3.- \space Nvidia \space Container \space Toolkit}$$
  - $${\color{orange}4.- \space Verify \space Docker \space Docker \space Installation}$$
- Docker Explanation
  - $${\color{royalblue}Animation}$$
  - $${\color{orange}Docker \space Process}$$
    - $${\color{royalblue}Animation}$$
    - $${\color{lightskyblue}Dockerfile}$$
    - $${\color{lightskyblue}Docker \space Image}$$
    - $${\color{lightskyblue}Docker \space Container}$$
      - $${\color{royalblue}Animation}$$
      - $${\color{blue}Parameters}$$
      - $${\color{royalblue}Animation}$$
- Run Proyect (Choose Environment)
  - $${\color{orange}Ubuntu \space Terminal}$$
  - $${\color{orange}Visual \space Studio \space Code \space (DevContainer)}$$
- Video

# How to Install

<p align="justify">
In order to execute and replicate the results the project was virtualized through docker being necessary only to install the tools to manage the virtual environment, also a dev container was made for visual studio code requiring less commands and a more friendly IDE to test and play on.
</p>

## $${\color{orange}1.- \space Docker \space Desktop}$$

<p align="justify">
Download and procced a to install <a href = "https://www.docker.com"> docker desktop </a> from the official website, once done enable the WSL2 connectivity currently on "<b>General</b>" and click on "<b>Use the WSL2 based engine</b>" or similar.
</p>

## $${\color{orange}2.- \space WSL2}$$

<p align="justify">

WSL2 is the windows subsystem for Linux, this will allow to integrate and use the kernel of many Linux distributions without need of virtual machine and being native on windows, to open it you must do the following path:
</p>

- Enable "**Virtual Machine Platform**" on windows features, could be found in "**Turn Windows features on or off**"
- Open windows PowerShell as an admin, then write "**wsl --install**"
- Write and login on Ubuntu distribution as a new user, then write "**sudo apt update && sudo apt upgrade -y**" to download all updates.

More information on <a href = "https://learn.microsoft.com/en-us/windows/wsl/install"> Microsoft </a> official website.

## $${\color{orange}3.- \space Nvidia \space Container \space Toolkit}$$

<p align="justify">
Nvidia Container Toolkit allows the containers made by Docker the use of all graphic cards naturally this gives the support for tensorflow to utilize a GPU on training models, however is also need it to use nvidia cuda platform to write parallel code and run it on the graphic card, this feature is applied with pycuda to makes faster the exploitation process of the metaheuristic algorithm.
</p>

Copy the following commands to download and install the files in the end restart the system.

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

More information on <a href = "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"> Nvidia </a> official website.

## $${\color{orange}4.- \space Verify \space Docker \space Docker \space Installation}$$

- Go to the ubuntu distribution download on WSL2 and type "**```docker --version```**" if everything is all right you should see the current version of docker you got on your system.
- Run the next docker image "**```docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi```**" if you see your GPU, Drivers and Cuda version everything was installed correct.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c1109a57-f7b6-4b87-98f8-50d7c4c668fe" alt="image">
</p>

# Docker Explanation

<p align="justify">
Docker is a tool to isolate virtual environments such Conda but giving more versatility and compatibility between machines, this means every single computer will run in the exact way what it is written using Docker, giving a longer lifespan to your software and in this particular case will let you to have multiple people working at the same project without any compatibility and version issues that is required to set new devices to work, besides it will give software reproducibility.
</p>

>[!IMPORTANT]
> At science and research environment it is significant to replicate the same experimental results.

<p align="justify">
Docker could be taken even further with the <b>Dev Container</b> extension which enable to export Docker to the Visual Studio Code IDE and even incorporated more custom settings such other extensions or personal QoS (Quality of Service) settings and their benefits compared virtual machines is the reuse of the kernel of the operative meanwhile the isolation between every single docker instance persist this allow to have an increased resource optimization and with other qualities from <b>Docker Compose</b> and <b>Docker Swarm</b> give productive software the ability to scale and adapt to requirements and demand.
</p>

https://github.com/user-attachments/assets/fc03ff05-4119-4dcc-9d09-b0900088cdf2

## $${\color{orange}Docker \space Process}$$

<p align="justify">
Docker takes multiples stages to concrete and create the virtual enviroments being Dockerfile, Image and Container.
</p>

https://github.com/user-attachments/assets/9f1a47e9-2737-4652-a91a-f8758c621232

### $${\color{lightskyblue}Dockerfile}$$

<p align="justify">
  Details all the setup instructions with the following keywords:
</p>

  - **FROM:** Written at the begging inherit configurations from a previous Docker Image found at Docker Hub, this save time reduce the docker file size and complexity, it is possible to make multiples FROM through a process called Multistage.
  - **WORKDIR:** Specifies the working directory in which all copy and data would be redirected.
  - **COPY:** Copies files between host and the virtual environment.
  - **RUN:** Allow to execute terminal commands such sudo update, upgrade or pip install.
  - **CMD:** Gives a predetermined routine when the environment is launched.
  - **EXPOUSE:** Enables a port to be used, for example 8888 with Jupyter Lab

### $${\color{lightskyblue}Docker \space Image}$$

<p align="justify">
Contain each and every single configuration from the Dockefile ready to run the virtual enviroment the image need to have an specific name with the following parts "user/proyect-name:tag":
</p>
  - user: Optional indicates the belong and user that created the image, it is need it when an image is uploaded to the cloud on Docker Hub to be used download from more users.
  - proyect-name: Encanpsulates the name of the repository.
  - tag: Specifies the purpouse or related information of their quialities.

The image is obtained with the following comand.

```
docker build -t {docker-image-name} {dockerfile-directory}
```

>[!TIP]
> The image could not use the cache of previous builds with --no-cache and compatibility with multiple platforms with --platform-linux/amd64,linux/arm64

### $${\color{lightskyblue}Docker \space Container}$$

<p align="justify">
Deploy an instance of an image with the virtual environment running with the specified qualities, multiple containers of the same image could be done and managed with Docker Compose, Docker Swarm and Kubernetes been able to segment the software on multiple individual containers according microservices software architecture.
</p>

https://github.com/user-attachments/assets/a3312a95-f922-4f9d-87ea-147b63ca3ddf

To run a container it is need it a couple of parameters

```
docker run -it --rm --gpus all -v {absolute-direction-host-machine:WORKDIR} --name {name-you-want-to-give-to-the-container} {docker image name}
```

- **it:** Allows and interactive terminal, it will allow to show feedback and logs from the terminal.
- **rm:** Removes the container once it is close.
- **gpus all:** Gives access to the container all the GPUs on your device using Docker Container Toolkit
- **v {absolute-direction-host-machine:WORKDIR}:** Creates a volume with the host machine in a specific folder, this makes it possible to storage any updated and created data through a container execution.
- **name:** Names the container.

>[!TIP]
> To run directly to the terminal and avoid the entrypoint or CMD instruction from de Dockerfile and Docker Image add "/bin/bash" to the end of the command "docker run -it ... /bin/bash".

https://github.com/user-attachments/assets/320895dc-a319-40eb-9ebc-47f5c833d2bb

# Run Proyect (Choose Environment)

<p align="justify">
Now you have two options to run the code, one being run it on the native WSL2 ubuntu terminal which is already installed or add one extension on visual studio to get a more comfortable develop environment.
</p>

## $${\color{orange}Ubuntu \space Terminal}$$

<p align="justify">
Run the next command and wait around 30 minutes for the docker image to be fully downloaded, <b>this will allow you to access the training results</b>.
</p>

```
docker run -it --rm --gpus all --name tensor pathinker/docker-tensorflow-wsl2:2.17.0
```

<p align="justify">
For now, you will need to write python3 and the complete route of files to execute them and other commands for editing, viewing the files, however all codes will be fully operational.
</p>

> [!CAUTION]
> Note: All changes made once the program is running will be lost due the missing of a volume that shares the data among the virtual environment and the host device.

## $${\color{orange}Visual \space Studio \space Code \space (DevContainer)}$$

<p align="justify">
To run it on Visual Studio Code you will need to search for extensions and type "<b><a href = "https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers">Dev Containers</a></b>" from Microsoft, wait until it is fully operational and then do the next shortcut "<b>Ctrl + Shift + P</b>", write and click on "<b>>Dev Containers: Rebuild and Reopen in Container</b>". Afterwards you will need to wait around the same time of the ubuntu terminal setup.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e5c8dd9f-792c-4b8e-9a95-9fd4aa7d2883" alt="image">
</p>

<p align="justify">
Using the dev container on Visual Studio Code will provide a few extensions to enable python debugging, fast code runner buttons, access to your git SSH keys to clone and modify the repository, zsh terminal, data persistence between changes and not opening the WSL2 or Ubuntu terminal.
</p>

> [!IMPORTANT]
> Docker engine must be running always if you want to reopen the project in both cases.

# Video

> [!NOTE]
> At this moment, the video is available only in Spanish. Due to current YouTube limitations, multiple audio tracks are not enabled on this channel. Supporting the content will help unlock this feature in the future, once the channel meets YouTube’s requirements.

[![Docket-Tensorflow-WSL2-Video](https://github.com/user-attachments/assets/9d67e5bd-0c52-49bc-8af9-990ebc75feeb)](https://youtu.be/acDmVYKnIVc)
