# 3D reconstruction of Autonomous Vehicle environment using Vectorized kernel Mixtures

This report presents an innovative approach for
3D reconstruction of autonomous vehicle environments using
raw point cloud data, leveraging the Vectorized Kernel Mixture
(VecKM [1]) method developed by Dehao Yuan, PhD scholar
at the University of Maryland. VecKM is distinguished by its
unparalleled efficiency, robustness to noise, and enhanced local
geometry encoding capabilities. Our methodology encompasses
data acquisition and preprocessing and environment reconstruc�tion through advanced feature extraction and deep learning
models. We validated the predicted normals against ground
truth normals using the PCPNet dataset, demonstrating VecKM’s
superior performance in terms of accuracy, computational cost,
memory efficiency, and robustness to noise. The results indicate
that VecKM significantly improves the processing and interpreta�tion of point cloud data, thereby advancing autonomous vehicle
technology by providing more accurate environment perception
and reliable navigation capabilities. This project sets a new
standard for point cloud data analysis in the field of autonomous
systems. We have also implemented this algorithm on the point
cloud data collected by simulating an autonomous vehicle in an
environment in CARLA simulator. After the data is collected,
using the VecKM we have predicted the normals of that Point
cloud data.

Please Download the PCPNet dataset from "https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/" and place it in the "VecKM" folder


To Train the model use
```bash
	- python main.py
```
To Validate the model select the test dataset list in the "val.py" file and use
```bash
 - python val.py
```
To reconstruct meshes of random objects, select the random object point cloud and normal files and use
```bash	
 - python reconstruct_meshes.py
```
To reconstruct CARLA town run
```bash
	- python reconstruct_carla_town.py
```
To run the simulation in CARLA

1. Install CARLA simulator in your device.

2. Go to the file "CarlaUE4.exe" and run it or start it. this is typically located at  	-"<path_to_carla_instalation>\CARLA_Latest\WindowsNoEditor\".

3. Open powerpoint shell as administrator and run the script "carla_data_acq.py" 
	-python run carla_data_acq.py

4. Make sure you wait before running the script until the carla server is fully initialized. (This may take some time). This will open a pygame window and the simulation can be visualized in the simulator

This will create a point cloud file "town3_carla_data.xyz", copy and paste it in the VecKM folder and run the "reconstruct_carla_town.py" to reconstruct the part of Carla town. 
