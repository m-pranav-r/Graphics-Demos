#include "parser.h"

#include "camera.h"
#include "shader.h"

int Smain() {
	try {
		ShaderHelper sh;
		//sh.init("fragment", Type::FRAG);
		//sh.readFile("./test.vert");
		sh.readFileGLSL("../shaders/sponza-pbr/shaderAll.frag");
		sh.addShaderDefinitionTerm("TESTTERM");
		sh.compileShaderToSPIRVAndCreateShaderModule();
		//sh.compileShaderFromFile("../shaders/pbr/shader.vert", "../shaders/pbr/verts.spv");

		std::cout << "done!\n";
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}