# Graphics Demos

## Present Demos:

### Sponza Test

A barebones render of the sponza scene rendered on a single thread. Repurposes the Galitef reader to process meshes. Intended to be used as a control to test against any further algorithms.

### demo-culling

A compute culling demo. Instance object data is sent to compute shader, which then generates draw commands based on visibility checks made against the view frustum. UI shows drawing data and the option is given to freeze frustum to better visualize the process.

## Libraries Used:

- Vulkan
- GLFW
- shaderc
- fastgltf
- glm
- dear Imgui