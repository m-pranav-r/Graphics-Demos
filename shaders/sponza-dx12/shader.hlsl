cbuffer UniformBuffer: register(b0)
{
    matrix model;
    matrix view;
    matrix proj;
    float3 camPos;
};
Texture2D BaseColorTexture : register(t0);
Texture2D MRTexture : register(t1);
Texture2D NormalTexture: register(t2);

SamplerState DefaultSampler: register(s0);

struct PSInput
{
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

PSInput VSMain(float3 position : POSITION, float3 normal: NORMAL, float4 tangent: TANGENT, float2 texCoord: TEXCOORD)
{
    PSInput result;

    result.position = mul(float4(position, 1.0), (model));
    result.texcoord = texCoord;

    return result;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    return BaseColorTexture.Sample(DefaultSampler, input.texcoord);
}