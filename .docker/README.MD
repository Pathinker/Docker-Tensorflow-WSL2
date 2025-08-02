# Docker Multistage

<p align="justify">
Docker gives the capacity to load multiple configurations from images, this is done to save time and lines to write the specific configurations, to make this possible each FROM that is typed to call an image is a single stage. Where the image is built and natively is given a natural number to each stage starting with 0 but giving a name is also possible.
</p>

<p align="justify">
After that it is only necessary to copy the specific configuration from the directory at the last stage, in this example a multistage is made to give python configuration to Ubuntu 22.04 operative system for a simple Cuda and cuDDN installation, instead of Debian 12 that the original image has.
</p>

>[!Warning]
> This process is only done when the image from tensorflow does not have the correct CUDA and cuDNN versions that need to detect and give support to GPU.

https://github.com/user-attachments/assets/944c2d9e-a2c0-4624-9c9d-d9f1dfa0f612