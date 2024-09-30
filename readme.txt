Team Members

- Abubakar Siddiq Palli | DirectoryID: absiddiq | UID: 120403422
- Gayatri Davuluri | DirectoryID: gayatrid | UID: 120304866
- Srividya Ponnada | DirectoryID: sponnada | UID: 120172748


Please Download the PCPNet dataset from "https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/" and place it in the "VecKM" folder


To Train the model use
	- python main.py

To Validate the model select the test dataset list in the "val.py" file and use
	- python val.py

To reconstruct meshes of random objects, select the random object point cloud and normal files and use
	- python reconstruct_meshes.py

To reconstruct CARLA town run
	- python reconstruct_carla_town.py

To run the simulation in CARLA


1. Install CARLA simulator in your device.

2. Go to the file "CarlaUE4.exe" and run it or start it. this is typically located at  	-"<path_to_carla_instalation>\CARLA_Latest\WindowsNoEditor\".

3. Open powerpoint shell as administrator and run the script "carla_data_acq.py" 
	-python run carla_data_acq.py

4. Make sure you wait before running the script until the carla server is fully initialized. (This may take some time). This will open a pygame window and the simulation can be visualized in the simulator

This will create a point cloud file "town3_carla_data.xyz", copy and paste it in the VecKM folder and run the "reconstruct_carla_town.py" to reconstruct the part of Carla town. 
