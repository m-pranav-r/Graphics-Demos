#include <d3d12.h>
#include <DirectXMath.h>
#include <wrl/client.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <directx/d3dx12.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <stdexcept>

using namespace DirectX;
using namespace Microsoft::WRL;

struct Vertex
{
	XMFLOAT3 position;
	XMFLOAT4 color;
};

inline void TIF(HRESULT hr)	//ThrowIfFailed
{
	if (FAILED(hr))
	{
		// Set a breakpoint on this line to catch Win32 API errors.
		throw std::exception();
	}
}

class DX12Test {
private:
	CD3DX12_VIEWPORT viewport;
	CD3DX12_RECT scissorRect;

	ComPtr<IDXGIFactory7> mainFactory;
	ComPtr<ID3D12Device2> mainDevice;
	ComPtr<ID3D12CommandQueue> mainCommandQueue;
	ComPtr<IDXGISwapChain4> mainSwapChain;
	ComPtr<ID3D12DescriptorHeap> rtvHeap;
	ComPtr<ID3D12Resource> renderTargets[5];
	ComPtr<ID3D12PipelineState> pipelineState;
	
	ComPtr<ID3D12CommandAllocator> commAllocator;
	ComPtr<ID3D12GraphicsCommandList> commList;

	ComPtr<ID3D12RootSignature> mainRootSignature;
	ComPtr<ID3D12Resource> vertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;

	ComPtr<ID3D12Fence> fence;
	UINT64 fenceValue;
	HANDLE fenceEvent;

	UINT frameIndex;
	UINT rtvDescriptorSize;

	UINT frameCount = 2;
	UINT width = 1280;
	UINT height = 720;

	GLFWwindow* window;

public:
	DX12Test() {};
	/*
	static DX12Test* getInst() {
		if (inst == nullptr) {
			inst = new DX12Test();
		}
		return inst;
	}
	
	static DX12Test* inst;
	*/

	bool g_IsInitialised = false;

	void InitWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		if (width == -1 || height == -1) {
			throw std::runtime_error("incorrectly set width and height parameters, exiting.\n");
		}
		window = glfwCreateWindow(width, height, "DX12 Test", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		//glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

		//glfwSetKeyCallback(window, keyCallback);
		//glfwSetCursorPosCallback(window, cursorPositionCallback);
		//glfwSetMouseButtonCallback(window, mouseButtonCallback);
		//glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
		//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}

	void OnInit() {
		InitWindow();
#if defined(_DEBUG)
		ComPtr<ID3D12Debug> spDebugController0;
		ComPtr<ID3D12Debug1> spDebugController1;
		TIF(D3D12GetDebugInterface(IID_PPV_ARGS(&spDebugController0)));
		spDebugController0->EnableDebugLayer();

		TIF(spDebugController0->QueryInterface(IID_PPV_ARGS(&spDebugController1)));
		spDebugController1->SetEnableGPUBasedValidation(true);
#endif
		//object that instantiates other COM objects
		TIF(CreateDXGIFactory1(IID_PPV_ARGS(&mainFactory)));

		//selecting the hardware device to run on
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

#if defined(_DEBUG)
		ID3D12InfoQueue* infoQueue = nullptr;
		mainDevice->QueryInterface(IID_PPV_ARGS(&infoQueue));

		infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
		infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
		infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, false);

		infoQueue->Release();
#endif

		//creating the command queue(?)
		D3D12_COMMAND_QUEUE_DESC queueDesc = {};
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT; //explore: what is 'direct' exactly

		TIF(mainDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&mainCommandQueue)));

		//creating the swapchain
		DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
		swapChainDesc.BufferCount = frameCount;
		swapChainDesc.BufferDesc.Width = width;
		swapChainDesc.BufferDesc.Height = height;
		swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;	//explore: what are the other modes
		swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;		//explore: what do the other swap modes do
		swapChainDesc.OutputWindow = glfwGetWin32Window(window);
		swapChainDesc.SampleDesc.Count = 1;								//explore: what
		swapChainDesc.Windowed = TRUE;

		ComPtr<IDXGISwapChain> swapChain;
		TIF(mainFactory->CreateSwapChain(mainCommandQueue.Get(), &swapChainDesc, &swapChain));
		swapChain.As(&mainSwapChain);

		TIF(mainFactory->MakeWindowAssociation(glfwGetWin32Window(window), DXGI_MWA_NO_ALT_ENTER));

		frameIndex = mainSwapChain->GetCurrentBackBufferIndex();

		//creating the descriptor heap(?)
		{
			D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
			rtvHeapDesc.NumDescriptors = frameCount;
			rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
			rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;		//explore: check other flags
			TIF(mainDevice->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)));

			rtvDescriptorSize = mainDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);//explore: !!!
		}

		//attaching the swapchain backbuffers to the heap(?)
		{
			CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());

			for (UINT n = 0; n < frameCount; n++) {
				TIF(mainSwapChain->GetBuffer(n, IID_PPV_ARGS(&renderTargets[n])));
				mainDevice->CreateRenderTargetView(renderTargets[n].Get(), nullptr, rtvHandle);
				rtvHandle.Offset(1, rtvDescriptorSize);
			}
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
	}

	void LoadAssets() {
		//creating an empty root signature(?)
		{
			CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
			//explore: what
			rootSignatureDesc.Init(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

			ComPtr<ID3DBlob> signature, error;
			TIF(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
			TIF(mainDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&mainRootSignature)));
		}

		{
			ComPtr<ID3DBlob> vertexShader, pixelShader;

#if defined(_DEBUG)
			UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
			UINT compileFlags = 0;
#endif
			//explore: what are these versions, maybe shader models?
			TIF(D3DCompileFromFile(L"../shaders/dx12/shaders.hlsl", nullptr, nullptr, "VSMain", "vs_5_0", compileFlags, 0, &vertexShader, nullptr));
			TIF(D3DCompileFromFile(L"../shaders/dx12/shaders.hlsl", nullptr, nullptr, "PSMain", "ps_5_0", compileFlags, 0, &pixelShader, nullptr));

			D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
				//explore: look at what they *exactly* mean
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
					"COLOR", 
					0, 
					DXGI_FORMAT_R32G32B32A32_FLOAT, 
					0, 
					12, 
					D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 
					0
				}
			};
			
			//make pipeline
			D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
			psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
			psoDesc.pRootSignature = mainRootSignature.Get();
			psoDesc.VS = { reinterpret_cast<UINT8*>(vertexShader->GetBufferPointer()), vertexShader->GetBufferSize() };
			psoDesc.PS = { reinterpret_cast<UINT8*>(pixelShader->GetBufferPointer()), pixelShader->GetBufferSize() };
			psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
			psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
			psoDesc.DepthStencilState.DepthEnable = FALSE;
			psoDesc.DepthStencilState.StencilEnable = FALSE;
			psoDesc.SampleMask = UINT_MAX;
			psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
			psoDesc.NumRenderTargets = 1;
			psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
			psoDesc.SampleDesc.Count = 1;
			TIF(mainDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState)));
		}

		//make command list
		TIF(mainDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commAllocator.Get(), pipelineState.Get(), IID_PPV_ARGS(&commList)));
		TIF(commList->Close());

		//create vertex buffer
		{
			Vertex triangleVertices[] = {
				{{0.0f, 0.25f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
				{{0.25f, -0.25f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
				{{-0.25f, -0.25f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
			};

			const UINT vertexBufferSize = sizeof(triangleVertices);

			//explore: try converting this to an actual real-world buffer copy
			CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
			auto desc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);
			TIF(
				mainDevice->CreateCommittedResource(
					&heapProps,
					D3D12_HEAP_FLAG_NONE,
					&desc,
					D3D12_RESOURCE_STATE_GENERIC_READ,
					nullptr,
					IID_PPV_ARGS(&vertexBuffer)
				)
			);

			UINT8* pVertexDatabegin;
			CD3DX12_RANGE readRange(0, 0);
			TIF(vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDatabegin)));
			memcpy(pVertexDatabegin, triangleVertices, sizeof(triangleVertices));
			vertexBuffer->Unmap(0, nullptr);

			vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
			vertexBufferView.StrideInBytes = sizeof(Vertex);
			vertexBufferView.SizeInBytes = vertexBufferSize;
		}

		{
			TIF(mainDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
			fenceValue = 1;

			fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
			if (!fenceEvent) {
				TIF(HRESULT_FROM_WIN32(GetLastError()));
			}
		}

		WaitForPreviousFrame();
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

	void OnUpdate() {

	}

	void OnRender() {
		PopulateCommandList();

		ID3D12CommandList* ppCommandLists[] = { commList.Get() };
		mainCommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

		TIF(mainSwapChain->Present(1, 0));			//explore: what do these mean

		WaitForPreviousFrame();
	}

	void PopulateCommandList() {
		TIF(commAllocator->Reset());

		TIF(commList->Reset(commAllocator.Get(), pipelineState.Get()));

		commList->SetGraphicsRootSignature(mainRootSignature.Get());
		commList->RSSetViewports(1, &viewport);
		commList->RSSetScissorRects(1, &scissorRect);
		
		auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
		commList->ResourceBarrier(1, &barrier);

		//explore: very sus
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, rtvDescriptorSize);
		commList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

		const float clearColor[] = { 0.4f, 0.2f, 0.0f, 1.0f };
		commList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
		commList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		commList->IASetVertexBuffers(0, 1, &vertexBufferView);
		commList->DrawInstanced(3, 1, 0, 0);

		barrier = CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
		commList->ResourceBarrier(1, &barrier);

		TIF(commList->Close());
	}

	void OnDestroy() {
		WaitForPreviousFrame();

		CloseHandle(fenceEvent);
	}
};

int main() {
	DX12Test test;
	test.OnInit();
	test.LoadAssets();
	while (true) {
		test.OnRender();
	}
}