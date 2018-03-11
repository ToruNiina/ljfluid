#ifndef LJ_CUDA_VECTOR_H
#define LJ_CUDA_VECTOR_H
#include <cmath>

namespace lj
{

__device__ __host__
inline float4 operator+(const float4& lhs, const float4& rhs)
{
    return make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, 0.0);
}

__device__ __host__
inline float4 operator+(const float4& lhs, const float rhs)
{
    return make_float4(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, 0.0);
}

__device__ __host__
inline float4 operator+(const float lhs, const float4& rhs)
{
    return make_float4(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, 0.0);
}

__device__ __host__
inline float4& operator+=(float4& lhs, const float4& rhs)
{
    lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z;
    return lhs;
}

__device__ __host__
inline float4& operator+=(float4& lhs, const float rhs)
{
    lhs.x += rhs; lhs.y += rhs; lhs.z += rhs;
    return lhs;
}



__device__ __host__
inline float4 operator-(const float4& lhs, const float4& rhs)
{
    return make_float4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, 0.0);
}

__device__ __host__
inline float4 operator-(const float4& lhs, const float rhs)
{
    return make_float4(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, 0.0);
}

__device__ __host__
inline float4 operator-(const float lhs, const float4& rhs)
{
    return make_float4(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, 0.0);
}

__device__ __host__
inline float4& operator-=(float4& lhs, const float4& rhs)
{
    lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z;
    return lhs;
}

__device__ __host__
inline float4& operator-=(float4& lhs, const float rhs)
{
    lhs.x -= rhs; lhs.y -= rhs; lhs.z -= rhs;
    return lhs;
}



__device__ __host__
inline float4 operator*(const float4& lhs, const float rhs)
{
    return make_float4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, 0.0);
}

__device__ __host__
inline float4 operator*(const float lhs, const float4& rhs)
{
    return make_float4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, 0.0);
}

__device__ __host__
inline float4& operator*=(float4& lhs, const float rhs)
{
    lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs;
    return lhs;
}



__device__ __host__
inline float4 operator/(const float4& lhs, const float rhs)
{
    return make_float4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, 0.0);
}

__device__ __host__
inline float4& operator/=(float4& lhs, const float rhs)
{
    lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs;
    return lhs;
}



__device__ __host__
inline float dot(const float4& lhs, const float4& rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__device__ __host__
inline float4 cross(const float4& lhs, const float4& rhs)
{
    return make_float4(lhs.y * rhs.z - lhs.z * rhs.y,
                       lhs.z * rhs.x - lhs.x * rhs.z,
                       lhs.x * rhs.y - lhs.y * rhs.x, 0.0);
}

__device__ __host__
inline float length_sq(const float4& lhs)
{
    return lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z;
}

__device__ __host__
inline float length(const float4& lhs)
{
    return sqrtf(length_sq(lhs));
}

} //lj
#endif// LJ_VECTOR
