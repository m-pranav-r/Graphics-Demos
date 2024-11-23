# Graphics Demos

## Present Demos:

### demo-culling

A compute culling demo. Instance object data is sent to compute shader, which then generates draw commands based on visibility checks made against the view frustum. UI shows drawing data and the option is given to freeze frustum to better visualize the process.

### demo-cascaded-shadow-mapping

A cascaded shadow mapping demo. Can choose number of splits and also includes toggles for PCF and for visualising chosen cascades. UI shows shadow maps drawn each frame among other things.

### demo-skeletal-anim

A simple demo to demonstrate skeletal animation on a basic skinned mesh.

### demo-quick-cone-mapping

A demo to demonstrate Quick Cone Mapping. In progress.

### sponza-brute

A barebones render of the sponza scene rendered on a single thread. Repurposes the Galitef reader to process meshes. Intended to be used as a control to test against any further algorithms.

### test-dx12

A smoke-testing DirectX12 application used for testing purposes.

### sponza-dx12

A DirectX12 port of the 'sponza-brute' demo.

## APIs Supported:
- DirectX12
- Vulkan


## Libraries Used:

- GLFW
- shaderc
- fastgltf
- glm
- dear Imgui