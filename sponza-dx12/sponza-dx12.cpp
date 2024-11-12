#include <d3d12.h>
//#include <DirectXMath.h>
#include <wrl/client.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <directx/d3dx12.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <stdexcept>
#include <algorithm>

#include "parser.h"
#include "camera.h"

//using namespace DirectX;
using namespace Microsoft::WRL;

struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec4 tangent;
	glm::vec2 texCoord;
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec3 camPos;
};

struct MaterialBufferObject {
	float roughnessFactor;
	float metallicFactor;
	glm::vec4 baseColorFactor;
};

const int MAX_FRAMES_IN_FLIGHT = 2;


struct DrawableHandle {
	ComPtr<ID3D12Resource> baseColorImage, metallicRoughnessImage, normalImage;
	//VkImageView baseColorImageView, metallicRoughnessImageView, normalImageView;
	//VkDeviceMemory baseColorMemory, metallicRoughnessMemory, normalMemory;
	std::vector<Vertex> vertices;
	ComPtr<ID3D12Resource> vertexBuffer, indexBuffer;
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
	D3D12_INDEX_BUFFER_VIEW indexBufferView;
	UINT descOffset;
	//using committed resources so no direct memory stuff for now
	//VkDeviceMemory vertexBufferMemory, indexBufferMemory; 
	size_t indices;

	bool isAlphaModeMask = false, hasNormal = false, hasMR = false, hasTangents = false;
};

Camera camera;

inline void TIF(HRESULT hr)
{
	if (FAILED(hr))
	{
		// Set a breakpoint on this line to catch Win32 API errors.
		throw std::exception();
	}
}

class DX12Sponza {
private:

	std::vector<Drawable> drawables;
	std::vector<DrawableHandle> drawableHandles;
	std::mutex drawableHandlesMutex;

	CD3DX12_VIEWPORT viewport;
	CD3DX12_RECT scissorRect;

	ComPtr<IDXGIFactory7> mainFactory;
	ComPtr<ID3D12Device2> mainDevice;
	ComPtr<ID3D12CommandQueue> mainCommandQueue;
	ComPtr<IDXGISwapChain4> mainSwapChain;
	ComPtr<ID3D12DescriptorHeap> rtvHeap, dsvHeap, srvHeap;
	ComPtr<ID3D12Resource> renderTargets[5];
	ComPtr<ID3D12Resource> dsvResource;
	ComPtr<ID3D12Resource> cbvResource;
	ComPtr<ID3D12PipelineState> pipelineState;

	CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle;
	UINT srvOffsetTracker = 0;
	std::mutex srvHandleMutex;

	ComPtr<ID3D12CommandAllocator> commAllocator;
	ComPtr<ID3D12GraphicsCommandList> commList;

	ComPtr<ID3D12RootSignature> mainRootSignature;

	D3D12_STATIC_SAMPLER_DESC defaultSampler;

	ComPtr<ID3D12Fence> fence;
	UINT64 fenceValue;
	HANDLE fenceEvent;
	std::mutex syncMutex;

	UINT frameIndex;
	UINT rtvDescriptorSize, dsvDescriptorSize, srvDescriptorSize;

	UINT width = 1280;
	UINT height = 720;

	std::chrono::steady_clock::time_point lastTime;

	void* constantBufferMapped;

	GLFWwindow* window;

public:
	DX12Sponza(uint32_t width, uint32_t height, std::vector<Drawable>& drawables) : width(width), height(height), drawables(drawables) {};

	void run() {
		initWindow();
		initDirectX();
		LoadAssets();
		mainLoop();
		cleanup();
	}

	bool g_IsInitialised = false;

	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		if (width == -1 || height == -1) {
			throw std::runtime_error("incorrectly set width and height parameters, exiting.\n");
		}
		window = glfwCreateWindow(width, height, "DX12 Test", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		//glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

		glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {camera.processGLFWKeyboardEvent(window, key, scancode, action, mods); });
		glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) { camera.processGLFWMouseEvent(window, x, y); });
		//glfwSetMouseButtonCallback(window, mouseButtonCallback);
		glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		camera.velocity = glm::vec3(0.0f, 0.0f, 0.0f);
		camera.position = glm::vec3(2.0f, 2.0f, 2.0f);
		camera.pitch = 0;
		camera.yaw = 0;
		camera.scalingFactor = 2;

		std::cout << "initWindow success...\n";
	}

	void initDirectX() {
#if defined(_DEBUG_EDITOR)
		ComPtr<ID3D12Debug> spDebugController0;
		ComPtr<ID3D12Debug1> spDebugController1;
		TIF(D3D12GetDebugInterface(IID_PPV_ARGS(&spDebugController0)));
		spDebugController0->EnableDebugLayer();

		TIF(spDebugController0->QueryInterface(IID_PPV_ARGS(&spDebugController1)));
		spDebugController1->SetEnableGPUBasedValidation(true);
#endif
		TIF(CreateDXGIFactory1(IID_PPV_ARGS(&mainFactory)));

		ComPtr<IDXGIAdapter1> hardwareAdapter1;
		ComPtr<IDXGIAdapter4> hardwareAdapter4;

		SIZE_T maxVRAM = 0;
		for (UINT i = 0; mainFactory->EnumAdapters1(i, &hardwareAdapter1) != DXGI_ERROR_NOT_FOUND; ++i) {
			DXGI_ADAPTER_DESC1 adapterDesc;
			hardwareAdapter1->GetDesc1(&adapterDesc);

			if ((adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 && SUCCEEDED(D3D12CreateDevice(hardwareAdapter1.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr)) && adapterDesc.DedicatedVideoMemory > maxVRAM) {
				maxVRAM = adapterDesc.DedicatedVideoMemory;
				TIF(hardwareAdapter1.As(&hardwareAdapter4));
			}
		}

		D3D12CreateDevice(hardwareAdapter4.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&mainDevice));

#if defined(_DEBUG_EDITOR)
		ID3D12InfoQueue* infoQueue = nullptr;
		mainDevice->QueryInterface(IID_PPV_ARGS(&infoQueue));

		infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
		infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
		infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, false);

		infoQueue->Release();
#endif

		D3D12_COMMAND_QUEUE_DESC queueDesc = {};
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

		TIF(mainDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&mainCommandQueue)));

		DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
		swapChainDesc.BufferCount = MAX_FRAMES_IN_FLIGHT;
		swapChainDesc.BufferDesc.Width = width;
		swapChainDesc.BufferDesc.Height = height;
		swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
		swapChainDesc.OutputWindow = glfwGetWin32Window(window);
		swapChainDesc.SampleDesc.Count = 1;
		swapChainDesc.SampleDesc.Quality = 0;
		swapChainDesc.Windowed = TRUE;

		ComPtr<IDXGISwapChain> swapChain;
		TIF(mainFactory->CreateSwapChain(mainCommandQueue.Get(), &swapChainDesc, &swapChain));
		swapChain.As(&mainSwapChain);

		TIF(mainFactory->MakeWindowAssociation(glfwGetWin32Window(window), DXGI_MWA_NO_ALT_ENTER));

		frameIndex = mainSwapChain->GetCurrentBackBufferIndex();

		//rtv/dsv heap creation
		{
			D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
			rtvHeapDesc.NumDescriptors = MAX_FRAMES_IN_FLIGHT;
			rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
			rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
			TIF(mainDevice->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)));

			rtvDescriptorSize = mainDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

			D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
			dsvHeapDesc.NumDescriptors = 1;
			dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
			dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
			TIF(mainDevice->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsvHeap)));

			dsvDescriptorSize = mainDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
		}

		{
			CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());

			for (UINT n = 0; n < MAX_FRAMES_IN_FLIGHT; n++) {
				TIF(mainSwapChain->GetBuffer(n, IID_PPV_ARGS(&renderTargets[n])));
				mainDevice->CreateRenderTargetView(renderTargets[n].Get(), nullptr, rtvHandle);
				rtvHandle.Offset(1, rtvDescriptorSize);
			}

			CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(dsvHeap->GetCPUDescriptorHandleForHeapStart());

			D3D12_HEAP_PROPERTIES dsHeapProps = {};
			dsHeapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
			dsHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
			dsHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
			dsHeapProps.CreationNodeMask = 0;
			dsHeapProps.VisibleNodeMask = 0;

			D3D12_RESOURCE_DESC dsResDesc = {};
			dsResDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
			dsResDesc.Alignment = 0;
			dsResDesc.Width = width;
			dsResDesc.Height = height;
			dsResDesc.DepthOrArraySize = 1;
			dsResDesc.MipLevels = 0;
			dsResDesc.Format = DXGI_FORMAT_D32_FLOAT;
			dsResDesc.SampleDesc.Count = 1;
			dsResDesc.SampleDesc.Quality = 0;
			dsResDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
			dsResDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;

			/*
			CD3DX12_RESOURCE_DESC dsResDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height, 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
			*/

			D3D12_CLEAR_VALUE clearValueDs = {};
			clearValueDs.Format = DXGI_FORMAT_D32_FLOAT;
			clearValueDs.DepthStencil.Depth = 1.0f;
			clearValueDs.DepthStencil.Stencil = 0;

			mainDevice->CreateCommittedResource(
				&dsHeapProps,
				D3D12_HEAP_FLAG_NONE,
				&dsResDesc,
				D3D12_RESOURCE_STATE_DEPTH_WRITE,
				&clearValueDs,
				IID_PPV_ARGS(&dsvResource)
			);

			D3D12_DEPTH_STENCIL_VIEW_DESC dsViewDesc = {};
			dsViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
			dsViewDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
			dsViewDesc.Flags = D3D12_DSV_FLAG_NONE;

			mainDevice->CreateDepthStencilView(dsvResource.Get(), &dsViewDesc, dsvHandle);
		}

		//srv heap creation
		{
			D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
			srvHeapDesc.NumDescriptors =  (1 + drawables.size()) * 3;
			srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
			srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;		//explore: check other flags
			TIF(mainDevice->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&srvHeap)));

			srvDescriptorSize = mainDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			srvHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(srvHeap->GetCPUDescriptorHandleForHeapStart());
		}

		//creating a command allocator
		TIF(mainDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commAllocator)));

		viewport.Height = 720.f;
		viewport.Width = 1280.f;
		viewport.TopLeftX = 0.f;
		viewport.TopLeftY = 0.f;
		viewport.MaxDepth = 1.0f;
		viewport.MinDepth = 1.0f;

		scissorRect.bottom = 720.f;
		scissorRect.top = 0.f;
		scissorRect.left = 0.f;
		scissorRect.right = 1280.f;

		{
			TIF(mainDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
			fenceValue = 1;

			fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
			if (!fenceEvent) {
				TIF(HRESULT_FROM_WIN32(GetLastError()));
			}
		}

		WaitForQueueMutexed();

		std::cout << "initDirectX success...\n";
	}

	void ProcessDrawable(Drawable drawable) {

		DrawableHandle currDrawableHandle;

		std::vector<Texture> textures = {
			drawable.mat.baseColorTex,
		};

		currDrawableHandle.isAlphaModeMask = drawable.mat.isAlphaModeMask;
		currDrawableHandle.hasMR = drawable.mat.hasMR;
		currDrawableHandle.hasNormal = drawable.mat.hasNormal;
		currDrawableHandle.hasTangents = drawable.hasTangents;

		if (currDrawableHandle.hasMR) textures.push_back(drawable.mat.metalRoughTex);
		else {
			return;
		}
		if (currDrawableHandle.hasNormal) textures.push_back(drawable.mat.normalTex);

		ComPtr<ID3D12CommandAllocator> tempAllocator;

		mainDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&tempAllocator));

		ComPtr<ID3D12GraphicsCommandList> tempCmdList;

		TIF(mainDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, tempAllocator.Get(), pipelineState.Get(), IID_PPV_ARGS(&tempCmdList)));

		tempCmdList->Close();

		srvHandleMutex.lock();

			currDrawableHandle.descOffset = srvOffsetTracker;
			srvOffsetTracker += 3;
			for (auto& texture : textures) {
			if (!texture.pixels) {
				throw std::runtime_error("failed to load texture image from memory!");
			}

			ComPtr<ID3D12Resource> textureImage;
			DXGI_FORMAT textureImageFormat;

			switch (texture.type) {
			case BASE:
			{
				textureImageFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
				break;
			}
			case METALLIC_ROUGHNESS: {
				textureImageFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
				break;
			}
			case NORMAL: {
				textureImageFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
				break;
			}
			}

			D3D12_RESOURCE_DESC textureDesc = {};
			textureDesc.MipLevels = 1;
			textureDesc.Format = textureImageFormat;
			textureDesc.Width = texture.texWidth;
			textureDesc.Height = texture.texHeight;
			textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
			textureDesc.DepthOrArraySize = 1;
			textureDesc.SampleDesc.Count = 1;
			textureDesc.SampleDesc.Quality = 0;
			textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

			TIF(mainDevice->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
				D3D12_HEAP_FLAG_NONE,
				&textureDesc,
				D3D12_RESOURCE_STATE_COPY_DEST,
				nullptr,
				IID_PPV_ARGS(&textureImage))
			);

			ComPtr<ID3D12Resource> textureUploadHeap;
			//textureUploadHeap->SetName(L"textureUploadHeap");

			const UINT64 imageSize = texture.texWidth * texture.texHeight * 4;

			TIF(mainDevice->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
				D3D12_HEAP_FLAG_NONE,
				&CD3DX12_RESOURCE_DESC::Buffer(imageSize),
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&textureUploadHeap))
			);

			D3D12_SUBRESOURCE_DATA textureData = {};
			textureData.pData = &texture.pixels[0];
			textureData.RowPitch = texture.texWidth * 4U;
			textureData.SlicePitch = textureData.RowPitch * texture.texHeight;

			tempCmdList->Reset(tempAllocator.Get(), pipelineState.Get());

			UpdateSubresources(tempCmdList.Get(), textureImage.Get(), textureUploadHeap.Get(), 0, 0, 1, &textureData);
			tempCmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(textureImage.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));

			D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
			srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			srvDesc.Format = textureDesc.Format;
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MipLevels = 1;

			mainDevice->CreateShaderResourceView(textureImage.Get(), &srvDesc, srvHandle);
			srvHandle.Offset(1, srvDescriptorSize);

			TIF(tempCmdList->Close());
			ID3D12CommandList* ppCommandLists[] = { tempCmdList.Get() };
			mainCommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

			switch (texture.type) {
			case BASE:
			{
				currDrawableHandle.baseColorImage = textureImage;
				break;
			}
			case METALLIC_ROUGHNESS: {
				currDrawableHandle.metallicRoughnessImage = textureImage;
				break;
			}
			case NORMAL: {
				currDrawableHandle.normalImage = textureImage;
				break;
			}
			}
			WaitForQueueMutexed();

		}
		
		srvHandleMutex.unlock();
		
		for (int j = 0; j < drawable.pos.size(); j++) {
			Vertex vertex{};

			vertex.pos = drawable.pos[j];
			vertex.normal = drawable.normals[j];
			if (drawable.hasTangents)vertex.tangent = drawable.tangents[j];
			else vertex.tangent = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
			vertex.texCoord = drawable.texCoords[j];

			currDrawableHandle.vertices.push_back(vertex);
		}

		drawable.pos.clear();
		drawable.pos.shrink_to_fit();
		drawable.normals.clear();
		drawable.normals.shrink_to_fit();
		drawable.tangents.clear();
		drawable.tangents.shrink_to_fit();
		drawable.texCoords.clear();
		drawable.texCoords.shrink_to_fit();

		//create vertex buffer
		{
			const UINT vertexBufferSize = sizeof(currDrawableHandle.vertices[0]) * currDrawableHandle.vertices.size();

			TIF(mainDevice->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
				D3D12_HEAP_FLAG_NONE,
				&CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize),
				D3D12_RESOURCE_STATE_COPY_DEST,
				nullptr,
				IID_PPV_ARGS(&currDrawableHandle.vertexBuffer))
			);

			ComPtr<ID3D12Resource> vertexUploadHeap;

			TIF(mainDevice->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
				D3D12_HEAP_FLAG_NONE,
				&CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize),
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&vertexUploadHeap))
			);

			D3D12_SUBRESOURCE_DATA vertexData = {};
			vertexData.pData = &currDrawableHandle.vertices[0];
			vertexData.RowPitch = sizeof(Vertex);
			vertexData.SlicePitch = vertexData.RowPitch * currDrawableHandle.vertices.size();

			tempAllocator->Reset();
			tempCmdList->Reset(tempAllocator.Get(), pipelineState.Get());

			UpdateSubresources(tempCmdList.Get(), currDrawableHandle.vertexBuffer.Get(), vertexUploadHeap.Get(), 0, 0, 1, &vertexData);
			tempCmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(currDrawableHandle.vertexBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));

			TIF(tempCmdList->Close());
			ID3D12CommandList* ppCommandLists[] = { tempCmdList.Get() };
			mainCommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

			WaitForQueueMutexed();

			currDrawableHandle.vertices.clear();
			currDrawableHandle.vertices.shrink_to_fit();

			currDrawableHandle.vertexBufferView.BufferLocation = currDrawableHandle.vertexBuffer->GetGPUVirtualAddress();
			currDrawableHandle.vertexBufferView.StrideInBytes = sizeof(Vertex);
			currDrawableHandle.vertexBufferView.SizeInBytes = vertexBufferSize;
		}

		//create index buffer
		{
			const UINT indexBufferSize = sizeof(drawable.indices[0]) * drawable.indices.size();

			TIF(mainDevice->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
				D3D12_HEAP_FLAG_NONE,
				&CD3DX12_RESOURCE_DESC::Buffer(indexBufferSize),
				D3D12_RESOURCE_STATE_COPY_DEST,
				nullptr,
				IID_PPV_ARGS(&currDrawableHandle.indexBuffer))
			);

			ComPtr<ID3D12Resource> indexUploadHeap;

			TIF(mainDevice->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
				D3D12_HEAP_FLAG_NONE,
				&CD3DX12_RESOURCE_DESC::Buffer(indexBufferSize),
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&indexUploadHeap))
			);

			D3D12_SUBRESOURCE_DATA indexData = {};
			indexData.pData = &drawable.indices[0];
			indexData.RowPitch = sizeof(uint32_t);
			indexData.SlicePitch = indexData.RowPitch * drawable.indices.size();

			tempAllocator->Reset();
			tempCmdList->Reset(tempAllocator.Get(), pipelineState.Get());

			UpdateSubresources(tempCmdList.Get(), currDrawableHandle.indexBuffer.Get(), indexUploadHeap.Get(), 0, 0, 1, &indexData);
			tempCmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(currDrawableHandle.indexBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));

			TIF(tempCmdList->Close());
			ID3D12CommandList* ppCommandLists[] = { tempCmdList.Get() };
			mainCommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

			currDrawableHandle.indices = drawable.indices.size();

			WaitForQueueMutexed();

			drawable.indices.clear();
			drawable.indices.shrink_to_fit();

			currDrawableHandle.indexBufferView.BufferLocation = currDrawableHandle.indexBuffer->GetGPUVirtualAddress();
			currDrawableHandle.indexBufferView.Format = DXGI_FORMAT_R32_UINT;
			currDrawableHandle.indexBufferView.SizeInBytes = indexBufferSize;
		}

		drawableHandlesMutex.lock();

			drawableHandles.push_back(currDrawableHandle);

		drawableHandlesMutex.unlock();

	}

	void processDrawables() {
		auto makeBuffersTask = std::async(std::launch::async,
			[&]() {
				std::for_each(std::execution::par,
				drawables.begin(),
				drawables.end(),
				[&](Drawable drawable) {
						ProcessDrawable(drawable);
					}
		);
			}
		);

		makeBuffersTask.wait();

		drawables.clear();
		drawables.shrink_to_fit();

		std::cerr << "\nAll drawables processed.\n";
	}

	void LoadAssets() {
		//creating an empty root signature(?)
		{
			D3D12_DESCRIPTOR_RANGE1 textureRange;
			textureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
			textureRange.NumDescriptors = 2;
			textureRange.BaseShaderRegister = 0;
			textureRange.RegisterSpace = 0;
			textureRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
			textureRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;//?

			D3D12_ROOT_PARAMETER1 vsTransform;
			vsTransform.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
			vsTransform.Descriptor = { 0, 0 };
			vsTransform.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

			D3D12_ROOT_PARAMETER1 psTextures;
			psTextures.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
			psTextures.DescriptorTable = { 1, &textureRange };
			psTextures.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

			D3D12_ROOT_SIGNATURE_FLAGS rootSigFlags =
				D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
				D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
				D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
				D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS;

			CD3DX12_STATIC_SAMPLER_DESC samplerDesc(
				0,
				D3D12_FILTER_MIN_MAG_MIP_LINEAR,
				D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
				D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
				D3D12_TEXTURE_ADDRESS_MODE_MIRROR
			);

			std::vector<D3D12_ROOT_PARAMETER1> rootParameters{ vsTransform, psTextures };

			//CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
			D3D12_ROOT_SIGNATURE_DESC1 rootSignatureDesc;
			rootSignatureDesc.NumParameters = static_cast<UINT>(rootParameters.size());
			rootSignatureDesc.pParameters = rootParameters.data();
			rootSignatureDesc.NumStaticSamplers = 1;
			rootSignatureDesc.pStaticSamplers = &samplerDesc;
			rootSignatureDesc.Flags = rootSigFlags;

			D3D12_VERSIONED_ROOT_SIGNATURE_DESC versionedRootSigDesc;
			versionedRootSigDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
			versionedRootSigDesc.Desc_1_1 = rootSignatureDesc;

			ComPtr<ID3DBlob> signature, error;
			if (FAILED(D3D12SerializeVersionedRootSignature(&versionedRootSigDesc, &signature, &error))) {
				if (error) {
					OutputDebugStringA((char*)error->GetBufferPointer());
				}
			}
			if (FAILED(mainDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&mainRootSignature)))) {
				if (error) {
					OutputDebugStringA((char*)error->GetBufferPointer());
				}
			}
		}

		{
			ComPtr<ID3DBlob> vertexShader, pixelShader;

#if defined(_DEBUG)
			UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
			UINT compileFlags = 0;
#endif
			ComPtr<ID3DBlob> shaderCompError;
			if (!SUCCEEDED(D3DCompileFromFile(L"../shaders/sponza-dx12/shader.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", compileFlags, 0, &vertexShader, &shaderCompError))) {
				if (shaderCompError) {
					OutputDebugStringA((char*)shaderCompError->GetBufferPointer());
					shaderCompError->Release();
				}
			}
			if (!SUCCEEDED(D3DCompileFromFile(L"../shaders/sponza-dx12/shader.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", compileFlags, 0, &pixelShader, &shaderCompError))) {
				if (shaderCompError) {
					OutputDebugStringA((char*)shaderCompError->GetBufferPointer());
					shaderCompError->Release();
				}
			}

			D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
				{
					"POSITION",										//SemanticName
					0,												//SemanticIndex
					DXGI_FORMAT_R32G32B32_FLOAT,					//Format
					0,												//InputSlot
					0,												//AlignedByteOffset
					D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,		//InputSlotClass
					0												//InstanceDataStepRate
				},
				{
					"NORMAL",
					0,
					DXGI_FORMAT_R32G32B32_FLOAT,
					0,
					12,
					D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
					0
				},
				{
					"TANGENT",
					0,
					DXGI_FORMAT_R32G32B32A32_FLOAT,
					0,
					24,
					D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
					0
				},
				{
					"TEXCOORD",
					0,
					DXGI_FORMAT_R32G32_FLOAT,
					0,
					40,
					D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
					0
				}
			};

			D3D12_DEPTH_STENCIL_DESC depthStencilDesc = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
			depthStencilDesc.DepthFunc = D3D12_COMPARISON_FUNC_LESS;

			//make pipeline
			D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
			psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
			psoDesc.pRootSignature = mainRootSignature.Get();
			psoDesc.VS = { reinterpret_cast<UINT8*>(vertexShader->GetBufferPointer()), vertexShader->GetBufferSize() };
			psoDesc.PS = { reinterpret_cast<UINT8*>(pixelShader->GetBufferPointer()), pixelShader->GetBufferSize() };
			psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
			psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
			psoDesc.DepthStencilState = depthStencilDesc;
			psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			psoDesc.NumRenderTargets = 1;
			psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
			psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
			psoDesc.SampleMask = UINT_MAX;
			psoDesc.SampleDesc.Count = 1;
			psoDesc.SampleDesc.Quality = 0;
			TIF(mainDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState)));
		}

		//make command list
		TIF(mainDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commAllocator.Get(), pipelineState.Get(), IID_PPV_ARGS(&commList)));
		TIF(commList->Close());

		processDrawables();

		//create constant buffer
		{
			const UINT constantBufferSize = (sizeof(UniformBufferObject) + 255) & ~255;

			TIF(
				mainDevice->CreateCommittedResource(
					&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
					D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(constantBufferSize),
					D3D12_RESOURCE_STATE_GENERIC_READ,
					nullptr,
					IID_PPV_ARGS(&cbvResource)
				)
			);

			D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
			cbvDesc.BufferLocation = cbvResource->GetGPUVirtualAddress();
			cbvDesc.SizeInBytes = constantBufferSize;
			mainDevice->CreateConstantBufferView(&cbvDesc, srvHandle);
			srvHandle.Offset(1, srvDescriptorSize);

			CD3DX12_RANGE readRange(0, 0);
			TIF(cbvResource->Map(0, &readRange, &constantBufferMapped));
		}

		std::cout << "LoadAssets success...\n";
	}

	void WaitForPreviousFrame() {
		const UINT64 l_fence = fenceValue;
		TIF(mainCommandQueue->Signal(fence.Get(), l_fence));
		fenceValue++;

		if (fence->GetCompletedValue() < l_fence) {
			TIF(fence->SetEventOnCompletion(l_fence, fenceEvent));
			WaitForSingleObject(fenceEvent, INFINITE);
		}

		frameIndex = mainSwapChain->GetCurrentBackBufferIndex();
	}

	void WaitForQueueMutexed() {
		syncMutex.lock();
		
		const UINT64 l_fence = fenceValue;
		TIF(mainCommandQueue->Signal(fence.Get(), l_fence));
		fenceValue++;

		if (fence->GetCompletedValue() < l_fence) {
			TIF(fence->SetEventOnCompletion(l_fence, fenceEvent));
			WaitForSingleObject(fenceEvent, INFINITE);
		}

		syncMutex.unlock();
	}

	void updateUniformBuffer(float deltaTime) {
		camera.update(deltaTime);

		UniformBufferObject ubo{};
		ubo.model = glm::scale(glm::mat4(1.0f), glm::vec3(0.000800000038));
		ubo.view = camera.getViewMatrix();
		ubo.proj = glm::perspectiveRH(glm::radians(45.0f), width / (float)height, 0.01f, 10.0f);
		//ubo.proj[3][3] *= -1;
		ubo.camPos = camera.position;

		auto mvp = ubo.proj * ubo.view * ubo.model;
		ubo.model = glm::transpose(mvp);

		memcpy(constantBufferMapped, &ubo, sizeof(UniformBufferObject));
	}

	void drawFrame() {
		PopulateCommandList();

		auto latestTime = std::chrono::high_resolution_clock::now();
		updateUniformBuffer(std::chrono::duration<float, std::chrono::seconds::period>(latestTime - lastTime).count());
		lastTime = latestTime;

		ID3D12CommandList* ppCommandLists[] = { commList.Get() };
		mainCommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

		TIF(mainSwapChain->Present(1, 0));

		WaitForPreviousFrame();
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}

		WaitForPreviousFrame();
	}

	void PopulateCommandList() {
		TIF(commAllocator->Reset());

		TIF(commList->Reset(commAllocator.Get(), pipelineState.Get()));

		commList->SetGraphicsRootSignature(mainRootSignature.Get());

		ID3D12DescriptorHeap* ppHeaps[] = { srvHeap.Get() };
		commList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

		commList->RSSetViewports(1, &viewport);
		commList->RSSetScissorRects(1, &scissorRect);
		
		commList->SetGraphicsRootConstantBufferView(0, cbvResource->GetGPUVirtualAddress());

		auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
		commList->ResourceBarrier(1, &barrier);

		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, rtvDescriptorSize);
		CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(dsvHeap->GetCPUDescriptorHandleForHeapStart());

		const float clearColor[] = { 0.4f, 0.2f, 0.0f, 1.0f };
		commList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
		commList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
		commList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

		commList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		for (size_t i = 0; i < drawableHandles.size(); i++) {
			commList->IASetVertexBuffers(0, 1, &drawableHandles[i].vertexBufferView);
			commList->IASetIndexBuffer(&drawableHandles[i].indexBufferView);
			CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(srvHeap->GetGPUDescriptorHandleForHeapStart());
			srvHandle.Offset(drawableHandles[i].descOffset, srvDescriptorSize);
			commList->SetGraphicsRootDescriptorTable(1, srvHandle);
			commList->DrawIndexedInstanced(drawableHandles[i].indices, 1, 0, 0, 0);
		}

		barrier = CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
		commList->ResourceBarrier(1, &barrier);

		TIF(commList->Close());
	}

	void cleanup() {
		WaitForPreviousFrame();

		CloseHandle(fenceEvent);
	}
};

int main() {
	try {
		GLTFParser sponzaParser;
		sponzaParser.parse_sponza("../models/Sponza/glTF/Sponza.gltf");
		DX12Sponza app(1280, 720, sponzaParser.drawables);
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}