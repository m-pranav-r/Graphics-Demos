#include "parser.h"

#ifndef STB_H
#define STB_H
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb_image.h>
#endif

std::string getMimeType(fastgltf::MimeType mimeType) {
	switch (mimeType)
	{
	case fastgltf::MimeType::None:
		return std::string("None");
		break;
	case fastgltf::MimeType::JPEG:
		return std::string("JPEG");
		break;
	case fastgltf::MimeType::PNG:
		return std::string("PNG");
		break;
	case fastgltf::MimeType::KTX2:
		return std::string("KTX2");
		break;
	case fastgltf::MimeType::DDS:
		return std::string("DDS");
		break;
	default:
		return std::string("unsupported MIME type!");
		break;
	}
}

bool Texture::load(fastgltf::Asset& asset, uint64_t textureIndex, TextureType texType, std::array<float, 4> factor, uint64_t texCoordIndex) {
	//std::cout << "trying to load texture of type " << texType;
	if (!asset.textures[textureIndex].imageIndex.has_value()) {
		std::cout << "...not found, marking as such.\n";
		return false;
	}
	fastgltf::Image& image = asset.images[asset.textures[textureIndex].imageIndex.value()];
	fastgltf::sources::BufferView bufferViewView = std::get<fastgltf::sources::BufferView>(image.data);
	fastgltf::BufferView& bufferView = asset.bufferViews[bufferViewView.bufferViewIndex];

	fastgltf::Buffer& buffer = asset.buffers[bufferView.bufferIndex];
	auto& byteView = std::get<fastgltf::sources::ByteView>(buffer.data);
	int requiredChannels = 4;
	pixels = stbi_load_from_memory((stbi_uc*)(byteView.bytes.data()), static_cast<int>(byteView.bytes.size()), &texWidth, &texHeight, &texChannels, requiredChannels);

	if (!pixels) {
		throw std::runtime_error("failed to load image into mem!\n");
	}
	this->factor = factor;
	this->type = texType;
	this->texCoordIndex = texCoordIndex;
	//std::cout << "... done!\n";
	return true;
}

bool Texture::load_from_loaded_texture_data(fastgltf::Asset& asset, uint64_t textureIndex, TextureType texType, std::array<float, 4> factor, uint64_t texCoordIndex) {
	//std::cout << "trying to load texture of type " << texType;
	if (!asset.textures[textureIndex].imageIndex.has_value()) {
		std::cout << "...not found, marking as such.\n";
		return false;
	}
	fastgltf::Image& image = asset.images[asset.textures[textureIndex].imageIndex.value()];
	fastgltf::sources::Array& dataArray = std::get<fastgltf::sources::Array>(image.data);

	int requiredChannels = 4;

	pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(dataArray.bytes.data()), static_cast<int>(dataArray.bytes.size()), &texWidth, &texHeight, &texChannels, requiredChannels);

	if (!pixels) {
		throw std::runtime_error("failed to load image into mem!\n");
	}
	this->factor = factor;
	this->type = texType;
	this->texCoordIndex = texCoordIndex;
	//std::cout << "... done!\n";
	return true;
}

bool Texture::load_from_file(fastgltf::Asset& asset, uint64_t textureIndex, TextureType texType, std::array<float, 4> factor, uint64_t texCoordIndex, std::string parentPath) {
	//std::cout << "trying to load texture of type " << texType;
	if (!asset.textures[textureIndex].imageIndex.has_value()) {
		std::cout << "...not found, marking as such.\n";
		return false;
	}
	fastgltf::Image& image = asset.images[asset.textures[textureIndex].imageIndex.value()];
	fastgltf::sources::URI& texturePath = std::get<fastgltf::sources::URI>(image.data);


	//		fastgltf::sources::Array &dataArray = std::get<fastgltf::sources::Array>(image.data);

	int requiredChannels = 4;

	pixels = stbi_load((parentPath + '/' + texturePath.uri.fspath().string()).c_str(), &texWidth, &texHeight, &texChannels, requiredChannels);

	//pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(dataArray.bytes.data()), static_cast<int>(dataArray.bytes.size()), &texWidth, &texHeight, &texChannels, requiredChannels);

	if (!pixels) {
		std::cout << stbi_failure_reason() << "\n" << (parentPath + '/' + texturePath.uri.fspath().string()).c_str() << "\n";
		throw std::runtime_error("failed to load image into mem!\n");
	}

	this->factor = factor;
	this->type = texType;
	this->texCoordIndex = texCoordIndex;
	//std::cout << "... done!\n";
	return true;
}

void GLTFParser::validateGLTF(fastgltf::Asset& asset) {
	fastgltf::Error validResult = fastgltf::validate(asset);

	switch (validResult)
	{
	case fastgltf::Error::None:
		break;
	case fastgltf::Error::InvalidPath:
		throw std::runtime_error("invalid path!");
		break;
	case fastgltf::Error::MissingExtensions:
		throw std::runtime_error("missing extension!");
		break;
	case fastgltf::Error::UnknownRequiredExtension:
		throw std::runtime_error("unknown required extension!");
		break;
	case fastgltf::Error::InvalidJson:
		throw std::runtime_error("invalid json!");
		break;
	case fastgltf::Error::InvalidGltf:
		throw std::runtime_error("invalid gltf!");
		break;
	case fastgltf::Error::InvalidOrMissingAssetField:
		throw std::runtime_error("invalid or missing asset field!");
		break;
	case fastgltf::Error::InvalidGLB:
		throw std::runtime_error("invalid glb!");
		break;
	case fastgltf::Error::MissingField:
		throw std::runtime_error("missing field!");
		break;
	case fastgltf::Error::MissingExternalBuffer:
		throw std::runtime_error("missing external buffer!");
		break;
	case fastgltf::Error::UnsupportedVersion:
		throw std::runtime_error("unsupported version!");
		break;
	case fastgltf::Error::InvalidURI:
		throw std::runtime_error("invalid uri!");
		break;
	case fastgltf::Error::InvalidFileData:
		throw std::runtime_error("invalid file data!");
		break;
	default:
		break;
	}
}

void GLTFParser::glTFError(fastgltf::Error error) {
	switch (error)
	{
	case fastgltf::Error::None:
		throw std::runtime_error("None");
		break;
	case fastgltf::Error::InvalidPath:
		throw std::runtime_error("InvalidPath");
		break;
	case fastgltf::Error::MissingExtensions:
		throw std::runtime_error("MissingExtensions");
		break;
	case fastgltf::Error::UnknownRequiredExtension:
		throw std::runtime_error("UnknownRequiredExtension");
		break;
	case fastgltf::Error::InvalidJson:
		throw std::runtime_error("InvalidJson");
		break;
	case fastgltf::Error::InvalidGltf:
		throw std::runtime_error("InvalidGltf");
		break;
	case fastgltf::Error::InvalidOrMissingAssetField:
		throw std::runtime_error("InvalidOrMissingAssetField");
		break;
	case fastgltf::Error::InvalidGLB:
		throw std::runtime_error("InvalidGLB");
		break;
	case fastgltf::Error::MissingField:
		throw std::runtime_error("MissingField");
		break;
	case fastgltf::Error::MissingExternalBuffer:
		throw std::runtime_error("MissingExternalBuffer");
		break;
	case fastgltf::Error::UnsupportedVersion:
		throw std::runtime_error("UnsupportedVersion");
		break;
	case fastgltf::Error::InvalidURI:
		throw std::runtime_error("InvalidURI");
		break;
	case fastgltf::Error::InvalidFileData:
		throw std::runtime_error("InvalidFileData");
		break;
	case fastgltf::Error::FailedWritingFiles:
		throw std::runtime_error("FailedWritingFiles");
		break;
	default:
		break;
	}
}

void GLTFParser::processPrimitive(fastgltf::Primitive &primitive, fastgltf::Asset &asset)
{
	Drawable currDrawable;
	//vertex data
	for (auto attrib : primitive.attributes) {
		//std::cout << "ACCESSOR DATA:\n" << attrib.first << "	" << attrib.second << "\n";
		if (attrib.first == "NORMAL") {
			auto& accessor = asset.accessors[attrib.second];
			currDrawable.normals.resize(accessor.count);

			std::size_t idx = 0;
			fastgltf::iterateAccessor<glm::vec3>(asset, accessor, [&](glm::vec3 index) {
				currDrawable.normals[idx++] = index;
				});
		}
		else if (attrib.first == "TANGENT") {
			currDrawable.hasTangents = true;
			auto& accessor = asset.accessors[attrib.second];
			currDrawable.tangents.resize(accessor.count);

			std::size_t idx = 0;
			fastgltf::iterateAccessor<glm::vec4>(asset, accessor, [&](glm::vec4 index) {
				currDrawable.tangents[idx++] = index;
				});

		}
		else if (attrib.first == "POSITION") {
			auto& accessor = asset.accessors[attrib.second];
			currDrawable.pos.resize(accessor.count);

			std::size_t idx = 0;
			fastgltf::iterateAccessor<glm::vec3>(asset, accessor, [&](glm::vec3 index) {
				currDrawable.pos[idx++] = index;
				});
		}
		else if (attrib.first == "TEXCOORD_0") {
			auto& accessor = asset.accessors[attrib.second];
			currDrawable.texCoords.resize(accessor.count);

			std::size_t idx = 0;
			fastgltf::iterateAccessor<glm::vec2>(asset, accessor, [&](glm::vec2 index) {
				currDrawable.texCoords[idx++] = index;
				});
		}
	}

	//indices data
	if (primitive.indicesAccessor.has_value()) {
		auto& accessor = asset.accessors[primitive.indicesAccessor.value()];
		currDrawable.indices.resize(accessor.count);

		std::size_t idx = 0;
		fastgltf::iterateAccessor<std::uint32_t>(asset, accessor, [&](std::uint32_t index) {
			currDrawable.indices[idx++] = index;
			});
	}

	//texture and material data
	auto& currMaterial = asset.materials[primitive.materialIndex.value()];

	currDrawable.mat.isAlphaModeMask = currMaterial.alphaMode == fastgltf::AlphaMode::Mask;

	if (currMaterial.pbrData.baseColorTexture.has_value()) {
		currDrawable.mat.hasBase = currDrawable.mat.baseColorTex.load_from_loaded_texture_data(
			asset,
			currMaterial.pbrData.baseColorTexture.value().textureIndex,
			TextureType::BASE,
			currMaterial.pbrData.baseColorFactor,
			currMaterial.pbrData.baseColorTexture.value().texCoordIndex
		);
	}

	if (currMaterial.pbrData.metallicRoughnessTexture.has_value()) {
		currDrawable.mat.hasMR = currDrawable.mat.metalRoughTex.load_from_loaded_texture_data(
			asset,
			currMaterial.pbrData.metallicRoughnessTexture.value().textureIndex,
			TextureType::METALLIC_ROUGHNESS,
			std::array<float, 4>{
			currMaterial.pbrData.metallicFactor,
				currMaterial.pbrData.roughnessFactor,
				0,
				0
		},
			currMaterial.pbrData.metallicRoughnessTexture.value().texCoordIndex
		);
	}

	if (currMaterial.normalTexture.has_value()) {
		currDrawable.mat.hasNormal = currDrawable.mat.normalTex.load_from_loaded_texture_data(
			asset,
			currMaterial.normalTexture.value().textureIndex,
			TextureType::NORMAL,
			std::array<float, 4>{currMaterial.normalTexture.value().scale},
			currMaterial.normalTexture.value().texCoordIndex
		);
	}

	drawablesMutex.lock();

	drawables.push_back(currDrawable);

	drawablesMutex.unlock();
}

void GLTFParser::parse_sponza(std::filesystem::path path) {
	fastgltf::Parser parser;

	//parentPath = path.parent_path().string();

	fastgltf::GltfDataBuffer data;
	data.loadFromFile(path);

	auto assetRef = parser.loadGltf(&data, path.parent_path(), fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages);

	if (auto error = assetRef.error(); error != fastgltf::Error::None) {
		glTFError(error);
		throw std::runtime_error("failed to load asset!");
	}
	//if (isDebugEnv) {
	validateGLTF(assetRef.get());
	//}

	fastgltf::Asset& asset = assetRef.get();

	std::cout << "successfully loaded & parsed gltf file!" << std::endl;

	uint16_t drawablesProcessedCount = 0;

	auto& scene = asset.scenes[asset.defaultScene.value()];
	auto& sponzaMainNode = asset.nodes[scene.nodeIndices[0]];
	auto& sponzaMainMesh = asset.meshes[sponzaMainNode.meshIndex.value()];

	// ONLY FOR SPONZA RENDERING!!!!
	auto loadTask = std::async(std::launch::async,
		[&]() {
			std::for_each(std::execution::par,
				sponzaMainMesh.primitives.begin(),
				sponzaMainMesh.primitives.end(),
				[&](fastgltf::Primitive primitive) {
					processPrimitive(primitive, asset);
				}
			);
		}
	);

	loadTask.wait();

	/*
	for (int i = 0; i < sponzaMainMesh.primitives.size(); i++) {
		processPrimitive(sponzaMainMesh.primitives[i], asset);
	}
	*/

	std::cerr << "\nAll drawables loaded.\n";
}

void GLTFParser::parse(std::filesystem::path path) {
	fastgltf::Parser parser;

	fastgltf::GltfDataBuffer data;
	data.loadFromFile(path);

	auto assetRef = parser.loadGltfBinary(&data, path.parent_path(), fastgltf::Options::None);
		
	if (auto error = assetRef.error(); error == fastgltf::Error::InvalidGLB) {
		assetRef = parser.loadGltf(&data, path.parent_path(), fastgltf::Options::None);

		if (auto error = assetRef.error(); error != fastgltf::Error::None) {
			throw std::runtime_error("failed to load asset!");
		}
	}

	validateGLTF(assetRef.get());

	fastgltf::Asset& asset = assetRef.get();

	Drawable model; //fix later

	std::cout << "successfully loaded & parsed gltf file!" << std::endl;

	for (auto& node : asset.scenes[0].nodeIndices) {

		auto currNode = asset.nodes[node];
		auto meshIndex = currNode.meshIndex;
		auto cameraIndex = currNode.cameraIndex;
		auto skinIndex = currNode.skinIndex;
		auto lightIndex = currNode.lightIndex;

		std::cout << "\nStarting to parse node " << node << " data, node has... \n";

		if (meshIndex.has_value()) std::cout << "\tmesh present...\n";
		if (cameraIndex.has_value()) std::cout << "\tcamera present...\n";
		if (skinIndex.has_value()) std::cout << "\tskin present...\n";
		if (lightIndex.has_value()) std::cout << "\tlight present...\n";

		model.transformData = std::get<fastgltf::TRS>(currNode.transform);

		std::cout << "...end of node listed node indices.\n";

		model.hasAnims = asset.animations.size() ? true : false;

		if (meshIndex.has_value()) {
			fastgltf::Mesh currMesh = asset.meshes[meshIndex.value()];

			for (auto& primitive : currMesh.primitives) {
				renderingMode = primitive.type;
				std::cout << "\n\tNode mesh primitive attributes present: " << primitive.attributes.size() << "\n";
				for (auto attrib : primitive.attributes) {
					std::cout << "\t\t" << attrib.first << "	" << attrib.second << "\n";
					if (attrib.first == "NORMAL") {
						auto& accessor = asset.accessors[attrib.second];
						model.normals.resize(accessor.count);

						std::size_t idx = 0;
						fastgltf::iterateAccessor<glm::vec3>(asset, accessor, [&](glm::vec3 index) {
							model.normals[idx++] = index;
							});
					}
					else if (attrib.first == "TANGENT") {
						auto& accessor = asset.accessors[attrib.second];
						model.tangents.resize(accessor.count);

						std::size_t idx = 0;
						fastgltf::iterateAccessor<glm::vec4>(asset, accessor, [&](glm::vec4 index) {
							model.tangents[idx++] = index;
							});
						model.hasTangents = true;

					}
					else if (attrib.first == "POSITION") {
						auto& accessor = asset.accessors[attrib.second];
						model.pos.resize(accessor.count);

						std::size_t idx = 0;
						fastgltf::iterateAccessor<glm::vec3>(asset, accessor, [&](glm::vec3 index) {
							model.pos[idx++] = index;
							});
					}
					else if (attrib.first == "TEXCOORD_0") {
						auto& accessor = asset.accessors[attrib.second];
						model.texCoords.resize(accessor.count);

						std::size_t idx = 0;
						fastgltf::iterateAccessor<glm::vec2>(asset, accessor, [&](glm::vec2 index) {
							model.texCoords[idx++] = index;
							});
					}
					else if (attrib.first == "JOINTS_0") {
						auto& accessor = asset.accessors[attrib.second];
						model.joints.resize(accessor.count);

						std::size_t idx = 0;
						fastgltf::iterateAccessor<glm::vec4>(asset, accessor, [&](glm::vec4 index) {
							model.joints[idx++] = index;
							});
					}
					else if (attrib.first == "WEIGHTS_0") {
						auto& accessor = asset.accessors[attrib.second];
						model.weights.resize(accessor.count);

						std::size_t idx = 0;
						fastgltf::iterateAccessor<glm::vec4>(asset, accessor, [&](glm::vec4 index) {
							model.weights[idx++] = index;
							});
					}
				}
			}
			fastgltf::Primitive& currPrim = currMesh.primitives[0];
			if (currPrim.indicesAccessor.has_value()) {
				auto& accessor = asset.accessors[currPrim.indicesAccessor.value()];
				model.indices.resize(accessor.count);

				std::size_t idx = 0;
				fastgltf::iterateAccessor<std::uint32_t>(asset, accessor, [&](std::uint32_t index) {
					model.indices[idx++] = index;
					});
			}

			if (currPrim.materialIndex.has_value()) {
				std::cout << "\tNode has material data, parsing...\n";

				auto& currMaterial = asset.materials[currPrim.materialIndex.value()];

				model.mat.baseColorTex.load(
					asset,
					currMaterial.pbrData.baseColorTexture.value().textureIndex,
					TextureType::BASE,
					currMaterial.pbrData.baseColorFactor,
					currMaterial.pbrData.baseColorTexture.value().texCoordIndex
				);

				model.mat.metalRoughTex.load(
					asset,
					currMaterial.pbrData.metallicRoughnessTexture.value().textureIndex,
					TextureType::METALLIC_ROUGHNESS,
					std::array<float, 4>{
					currMaterial.pbrData.metallicFactor,
						currMaterial.pbrData.roughnessFactor,
						0,
						0
				},
					currMaterial.pbrData.metallicRoughnessTexture.value().texCoordIndex
				);

				model.mat.normalTex.load(
					asset,
					currMaterial.normalTexture.value().textureIndex,
					TextureType::NORMAL,
					std::array<float, 4>{currMaterial.normalTexture.value().scale},
					currMaterial.normalTexture.value().texCoordIndex
				);

				if (currMaterial.emissiveTexture.has_value()) {
					model.mat.hasEmissive = model.mat.emissiveTex.load(
						asset,
						currMaterial.emissiveTexture.value().textureIndex,
						TextureType::EMISSIVE,
						std::array<float, 4>{
						currMaterial.emissiveFactor[0],
							currMaterial.emissiveFactor[1],
							currMaterial.emissiveFactor[2],
							1
					},
						currMaterial.emissiveTexture.value().texCoordIndex
					);

				}

				if (currMaterial.occlusionTexture.has_value()) {
					model.mat.hasOcclusion = model.mat.occlusionTex.load(
						asset,
						currMaterial.occlusionTexture.value().textureIndex,
						TextureType::OCCLUSION,
						std::array<float, 4>{currMaterial.occlusionTexture.value().strength},
						currMaterial.occlusionTexture.value().texCoordIndex
					);
				}
			}

			std::cout << "\t...mesh parsing done.\n";
		}

		if (skinIndex.has_value()) {
			std::cout << "\n\tNode has skin data, parsing...";
			fastgltf::Skin currSkin = asset.skins[skinIndex.value()];

			auto numJoints = currSkin.joints.size();
			model.skeleton.joints.resize(numJoints);

			auto& accessor = asset.accessors[currSkin.inverseBindMatrices.value()];
			model.skeleton.invBindMatrices.resize(accessor.count);

			std::size_t idx = 0;
			fastgltf::iterateAccessor<glm::mat4>(asset, accessor, [&](glm::mat4 index) {
				model.skeleton.invBindMatrices[idx++] = index;
				}
			);

			for (auto& currJointIdx : currSkin.joints) {
				Joint joint;
				auto currJoint = asset.nodes[currJointIdx];
				
				if (currJoint.transform.index() == 0) {
					auto TRS = std::get<fastgltf::TRS>(currJoint.transform);
					glm::mat4 matrix = glm::mat4(1.0f);
					matrix = glm::scale(matrix, glm::make_vec3(TRS.scale.data()));
					matrix = glm::toMat4(glm::quat(TRS.rotation[3], TRS.rotation[0], TRS.rotation[1], TRS.rotation[2])) * matrix;	//gltf gives x,y,z,w and glm expects w,x,y,z
					matrix = glm::translate(matrix, glm::make_vec3(TRS.translation.data()));
					joint.transform = matrix;
				} else {
					auto matrix = std::get<fastgltf::Node::TransformMatrix>(currJoint.transform);
					joint.transform = glm::make_mat4(matrix.data());
				}
				joint.globalTransform = glm::mat4(1.0f);

				joint.children.reserve(currJoint.children.size());
				for (size_t i = 0; i < currJoint.children.size(); i++)
					joint.children.push_back(currJoint.children[i]);	//i hate this

				model.skeleton.joints[currJointIdx - 1] = joint;
			}

			for (auto& currAnim : asset.animations) {
				for (auto& channel : currAnim.channels) {
					fastgltf::AnimationSampler currSampler = currAnim.samplers[channel.samplerIndex];
					
					Animation anim;
					anim.type =
						channel.path == fastgltf::AnimationPath::Translation ? AnimType::TRANS :
						channel.path == fastgltf::AnimationPath::Rotation ? AnimType::ROT :
						channel.path == fastgltf::AnimationPath::Scale ? AnimType::SCALE : AnimType::WEIGHTS;
					
					anim.interpType =
						currSampler.interpolation == fastgltf::AnimationInterpolation::Linear ? AnimInterpType::LINEAR :
						currSampler.interpolation == fastgltf::AnimationInterpolation::Step ? AnimInterpType::STEP : AnimInterpType::CUBICSPLINE;
					
					anim.jointIdx = channel.nodeIndex.value();

					{
						auto animInputAccessor = asset.accessors[currSampler.inputAccessor];
						std::size_t idx = 0;
						anim.keyframeTimings.resize(animInputAccessor.count);
						fastgltf::iterateAccessor<float>(asset, animInputAccessor, [&](float index) {
							anim.keyframeTimings[idx++] = index;
							}
						);
					}

					{
						auto animOutputAccessor = asset.accessors[currSampler.outputAccessor];
						std::size_t idx = 0;
						if (anim.type == AnimType::ROT) {
							anim.keyframeValues.emplace<1>(std::vector<glm::vec4>());
							auto vecPointer = &std::get<1>(anim.keyframeValues);
							vecPointer->resize(animOutputAccessor.count);
							fastgltf::iterateAccessor<glm::vec4>(asset, animOutputAccessor, [&](glm::vec4 index) {
								vecPointer->operator[](idx++) = index;
								}
							);

						} else {
							anim.keyframeValues.emplace<0>(std::vector<glm::vec3>());
							auto vecPointer = &std::get<0>(anim.keyframeValues);
							vecPointer->resize(animOutputAccessor.count);
							fastgltf::iterateAccessor<glm::vec3>(asset, accessor, [&](glm::vec3 index) {
								vecPointer->operator[](idx++) = index;
								}
							);
						}
					}

					model.skeleton.animations.push_back(anim);
				}
			}

			std::cout << " parsed\n";
		}
	}

	drawables.push_back(model);
}