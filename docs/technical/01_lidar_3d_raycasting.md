# 3D LiDAR Raycasting Engine

The primary method for determining the precise geospatial location of detected infrastructure (electrical boxes / *firidas*) is a highly accurate 3D LiDAR raycasting engine. This approach fuses panoramic imagery bounding boxes with dense LiDAR point clouds to extract real-world coordinates.

Here is a step-by-step breakdown of how the engine works:

## 1. Unprojecting the 2D Bounding Box
When the YOLOv8 model detects a *firida* in a camera frame, it provides a 2D bounding box (X and Y pixel coordinates). The engine extracts the **centroid** of this box. Using the specific intrinsic lens parameters (focal length and principal point) of the Ladybug 5 camera lens, this 2D pixel is mathematically "unprojected" into a 3D unit direction vector in the camera's local space.

## 2. World Space Translation & Bearing
The local camera vector must be translated into the real world. We take the vehicle's heading (provided by the IMU/GPS telemetry), add the known physical mount angle of the specific camera lens on the vehicle's roof, and apply the pixel's local yaw offset. This gives us the **True Geographic Bearing** of the ray being cast from the lens.

## 3. Altitude Calibration (Geoid Undulation)
A critical challenge in mobile mapping is that GPS logs altitude in the **WGS84 Ellipsoidal datum**, while LiDAR point clouds typically record physical elevation in the **Orthometric datum** (e.g., above the Black Sea).

To prevent our raycast from missing the point cloud entirely, the system performs a dynamic, localized calibration:
- It samples the vehicle's path.
- It searches the LiDAR cloud directly beneath the vehicle to find the physical road surface.
- It calculates the exact gap (the Geoid Undulation) between the GPS altitude and the LiDAR road surface.
- It subtracts this undulation from the camera's origin, aligning the raycast perfectly with the point cloud's Z-axis.

## 4. Cylindrical KDTree Intersection
With a calibrated origin point and a precise 3D vector, the system casts a virtual "ray" into the LiDAR point cloud.

Because point clouds are sparse networks of discrete points rather than solid surfaces, a mathematical line will rarely hit a point exactly. Instead, the engine uses a **Cylindrical Raycast**:
- We load the LiDAR data into a highly efficient **KDTree** data structure.
- We step along the ray in 1-meter increments up to a maximum distance.
- At each step, we query the KDTree for any LiDAR points falling within a tight cylinder (e.g., 20cm radius) wrapped around the ray.

## 5. Sub-Centimeter Extraction
When the ray penetrates a dense cluster of points inside the cylinder, we have hit the physical object. The system aggregates the points at the impact site and returns their centroid. 

This yields a highly accurate, real-world **X, Y, and Z coordinate** in the native Stereo70 projection (EPSG:3844), capturing not just the geographic footprint of the *firida*, but its exact elevation as well.
