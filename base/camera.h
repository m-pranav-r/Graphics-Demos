#pragma once

#ifndef GLM_H
#define GLM_H
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/quaternion.hpp>
#endif

#include <GLFW/glfw3.h>

#include <iostream>

class Camera {
public:
	glm::vec3 velocity;
	glm::vec3 position;
	float pitch = 0.f;
	float yaw = 0.f;

	float scalingFactor = 10;

	double prevFrameXPos = -1, prevFrameYPos = -1;

	glm::mat4 getViewMatrix();

	glm::mat4 getRotationMatrix();

	void processGLFWKeyboardEvent(GLFWwindow* window, int key, int scancode, int action, int mods);

	void processGLFWMouseEvent(GLFWwindow* window, double x, double y);

	void update(float deltaTime);
};