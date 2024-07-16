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

void Camera::processGLFWKeyboardEvent(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		if (key == GLFW_KEY_W) velocity.z = -1;
		if (key == GLFW_KEY_A) velocity.x = -1;
		if (key == GLFW_KEY_S) velocity.z = 1;
		if (key == GLFW_KEY_D) velocity.x = 1;
		if (key == GLFW_KEY_E) velocity.y = 1;
		if (key == GLFW_KEY_Q) velocity.y = -1;
		if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, 1);
	}

	if (action == GLFW_RELEASE) {
		if (key == GLFW_KEY_W) velocity.z = 0;
		if (key == GLFW_KEY_A) velocity.x = 0;
		if (key == GLFW_KEY_S) velocity.z = 0;
		if (key == GLFW_KEY_D) velocity.x = 0;
		if (key == GLFW_KEY_E) velocity.y = 0;
		if (key == GLFW_KEY_Q) velocity.y = 0;
	}
}

void Camera::processGLFWMouseEvent(GLFWwindow* window, double x, double y) {
	if (prevFrameXPos == -1) {
		prevFrameXPos = x;
		prevFrameYPos = y;
	}
	yaw += (x - prevFrameXPos) / 2000.f;
	pitch -= (y - prevFrameYPos) / 2000.f;

	prevFrameXPos = x;
	prevFrameYPos = y;
}

void Camera::update() {
	glm::mat4 cameraRotation = getRotationMatrix();
	position += glm::vec3(cameraRotation * glm::vec4(velocity * 0.005f, 0.f));
}