cbuffer constantBufferPlaceholder : register(b0)
{
    uint value;
};

struct PSInput
{
    float4 position : SV_POSITION;
};

PSInput VSMain(float3 position : POSITION, float3 normal: NORMAL, float4 tangent: TANGENT, float2 texCoord: TEXCOORD)
{
    PSInput result;

    result.position = float4(position, 1.0);

    return result;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    return input.position;
}