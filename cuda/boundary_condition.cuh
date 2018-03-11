#ifndef LJ_BOUNDARY_CONDITION
#define LJ_BOUNDARY_CONDITION
#include <cuda/vec.cuh>

namespace lj
{

struct periodic_boundary
{
    float4 upper;
    float4 lower;
    float4 width;
    float4 half;
};

__host__
inline periodic_boundary make_boundary(const float4& lower, const float4& upper)
{
    return periodic_boundary{upper, lower, upper - lower, (upper - lower) / 2};
}

__device__ __host__
inline bool is_inside_of(float4 pos, const periodic_boundary& b) noexcept
{
    if     (pos.x <  b.lower.x){return false;}
    else if(pos.x >= b.upper.x){return false;}
    if     (pos.y <  b.lower.y){return false;}
    else if(pos.y >= b.upper.y){return false;}
    if     (pos.z <  b.lower.z){return false;}
    else if(pos.z >= b.upper.z){return false;}
    return true;
}

__device__ __host__
inline float4 adjust_direction(float4 pos, const periodic_boundary& b) noexcept
{
         if(pos.x <  -b.half.x){pos.x += b.width.x;}
    else if(pos.x >=  b.half.x){pos.x -= b.width.x;}
         if(pos.y <  -b.half.y){pos.y += b.width.y;}
    else if(pos.y >=  b.half.y){pos.y -= b.width.y;}
         if(pos.z <  -b.half.z){pos.z += b.width.z;}
    else if(pos.z >=  b.half.z){pos.z -= b.width.z;}
    return pos;
}

__device__ __host__
inline float4 adjust_position(float4 pos, const periodic_boundary& b) noexcept
{
         if(pos.x <  b.lower.x){pos.x += b.width.x;}
    else if(pos.x >= b.upper.x){pos.x -= b.width.x;}
         if(pos.y <  b.lower.y){pos.y += b.width.y;}
    else if(pos.y >= b.upper.y){pos.y -= b.width.y;}
         if(pos.z <  b.lower.z){pos.z += b.width.z;}
    else if(pos.z >= b.upper.z){pos.z -= b.width.z;}
    return pos;
}

} // lj
#endif// LJ_BOUNDARY_CONDITION
