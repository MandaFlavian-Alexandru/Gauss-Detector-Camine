# Planar Fallback Geometry

While the 3D LiDAR raycasting engine is our primary and most precise method for extracting geospatial coordinates, real-world conditions are rarely perfect. Sometimes a *firida* is visible in the panoramic camera but cannot be intersected in the LiDAR point cloud.

This typically happens for two reasons:
1. **Occlusion:** A passing truck, tree branch, or pedestrian blocks the laser pulses from hitting the object, creating a "shadow" in the point cloud.
2. **Sparsity:** The object is too far away, or the material of the object absorbed the LiDAR pulses, leaving too few points for the KDTree intersection to trigger a confident hit.

To ensure that no detections are lost due to missing LiDAR data, the system seamlessly transitions to a robust **Planar Fallback** algorithm.

## How the Planar Fallback Works

When the cylindrical raycast travels its maximum distance without hitting the point cloud, the system shifts from 3D intersection math to 2D flat-earth trigonometry.

### 1. The Flat-Earth Assumption
The algorithm assumes that the detected object is resting on the ground, and that the ground between the vehicle and the object is relatively flat. 

### 2. Trigonometric Distance Estimation
Using the 3D unit direction vector generated during the raycasting phase, the engine analyzes the **pitch** (vertical angle) of the ray looking down from the camera lens to the bounding box centroid.

Because we know the exact physical height of the camera lens above the road surface (e.g., 2.45 meters), we have a right-angle triangle where:
- The "opposite" side is the camera height.
- The angle is the pitch of the unprojected pixel.

Using basic trigonometry (`Distance = Camera Height / tan(Pitch)`), the engine calculates the estimated distance along the ground from the vehicle to the object.

### 3. Coordinate Projection
Once the ground distance is estimated, the system uses the True Geographic Bearing (calculated previously) to project a point out from the vehicle's GPS coordinates along that bearing, at the estimated distance.

### 4. Output
This yields an estimated **X and Y coordinate** in the Stereo70 projection. Because we are relying on a flat-earth assumption rather than measuring the physical object, the `Z` (altitude) coordinate is inherently estimated based on the vehicle's altitude.

Detections mapped using this fallback method are explicitly flagged in the final database (`lidar_hit: false`), allowing GIS operators and reviewers to instantly distinguish between sub-centimeter LiDAR strikes and estimated planar locations.
