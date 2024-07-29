#include "camera.h"

glm::mat4 Camera::getViewMatrix() {
	glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.0f), position);
	glm::mat4 cameraRotation = getRotationMatrix();
	return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::getRotationMatrix() {
	glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3{ 1.f, 0.f, 0.f });
	glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3({ 0.f, -1.f, 0.f }));

	return glm::mat4_cast(yawRotation) * glm::mat4_cast(pitchRotation);
}

inline int keytoArrayElement(int key) {
	return
		key == GLFW_KEY_W || key == GLFW_KEY_S ? 2 :
		key == GLFW_KEY_A || key == GLFW_KEY_D ? 0 :
		1;
}

inline int keyToSign(int key) {
	return
		key == GLFW_KEY_W || key == GLFW_KEY_A || key == GLFW_KEY_Q ? -1 : 1;
}

void Camera::processGLFWKeyboardEvent(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		if (key == GLFW_KEY_W) velocity.z = -1;
		else if (key == GLFW_KEY_A) velocity.x = -1;
		else if (key == GLFW_KEY_S) velocity.z = 1;
		else if (key == GLFW_KEY_D) velocity.x = 1;
		else if (key == GLFW_KEY_E) velocity.y = 1;
		else if (key == GLFW_KEY_Q) velocity.y = -1;
		else if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, 1);
	}else if (action == GLFW_RELEASE) {
		if (key == GLFW_KEY_W) velocity.z = 0;
		else if (key == GLFW_KEY_A) velocity.x = 0;
		else if (key == GLFW_KEY_S) velocity.z = 0;
		else if (key == GLFW_KEY_D) velocity.x = 0;
		else if (key == GLFW_KEY_E) velocity.y = 0;
		else if (key == GLFW_KEY_Q) velocity.y = 0;
	}
}

void Camera::processGLFWMouseEvent(GLFWwindow* window, double x, double y) {
	//std::cout << "MOUSEEVENT CALLED\n";
	if (prevFrameXPos == -1) {
		prevFrameXPos = x;
		prevFrameYPos = y;
	}

	yaw += (x - prevFrameXPos) / 2000.f;
	pitch -= (y - prevFrameYPos) / 2000.f;

	prevFrameXPos = x;
	prevFrameYPos = y;
}

void Camera::update(float deltaTime) {
	glm::mat4 cameraRotation = getRotationMatrix();
	position += deltaTime * scalingFactor * glm::vec3(cameraRotation * glm::vec4(velocity, 0.f));
}