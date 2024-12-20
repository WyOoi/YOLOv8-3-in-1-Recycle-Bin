# YOLOv8-3-in-1-Recycle-Bin
This project using Lego Mindstorm EV3 communicate with Raspberry Pi equipped with Rapsberry pi Camera to run the recyclable waste detection using YOLOv8. The project successfully won the Silver Award in the National Robotics Competition State Level Future Innovator Category. This project mainly focus on sorting the recyclable waste to the bin using the image proccessing with raspberry pi camera using YOLOv8

<div align="center">
   <img src="pic/Png1.png" alt="png">
</div>

## Table of Contents
1. [Sofware Used](#Software_Used)
2. [Install Ultralytics](#Install_Ultralytics)
3. [Features](#features)
4. [Documentation](#documentation)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgments](#acknowledgments)
8. [Website Address](#Website_address)

## Download & Installation

### Step 1: Install the software used
1. Anaconda Prompt (Miniconda3)[Link text](https://www.anaconda.com/download)
2. Lego Mindstorm EV3 Programming software (EV3-G).[Link text](https://education.lego.com/en-us/downloads/retiredproducts/mindstorms-ev3-lab/software/).

### Step 2: Create virtual environment in Anaconda
1. command create new environment:
   ```python
   conda create --name myenv python=3.9

2. Enter the environment:
   ```python
   conda activate myenv

3. *Exit the environment:*
   ```python
   conda deactivate

### Step 3: Download this github respiratoty to your laptop or using the step below
1. Clone the repository:
   ```bash
   git clone https://github.com/WyOoi/YOLOv8-3-in-1-Recycle-Bin.git
   
### Install Library
1. Install library for ultralytics
   ```python
   pip3 install ultralytics

## Run the code
1. Command to run the code
   ```python
      python test.py
   
## License
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Copyright (C) 2024 WyOoi
-   Licensed under the GNU General Public License v3.0 (the "License").
-   You may not use this project or any file except in compliance with the License.
-   You may obtain a copy of the License at [https://www.gnu.org/licenses/#GPL](https://www.gnu.org/licenses/#GPL).
