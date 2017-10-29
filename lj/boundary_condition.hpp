#ifndef LJ_BOUNDARY_CONDITION
#define LJ_BOUNDARY_CONDITION
#include "vector.hpp"

namespace lj
{

template<typename T>
vector<T> adjust_direction(vector<T> pos,
        const vector<T>& size, const vector<T>& size_half)
{
         if(pos.x < -size_half.x) pos.x += size.x;
    else if(pos.x >  size_half.x) pos.x -= size.x;
         if(pos.y < -size_half.y) pos.y += size.y;
    else if(pos.y >  size_half.y) pos.y -= size.y;
         if(pos.z < -size_half.z) pos.z += size.z;
    else if(pos.z >  size_half.z) pos.z -= size.z;
    return pos;
}

template<typename T>
vector<T> adjust_position(vector<T> pos,
        const vector<T>& upper, const vector<T>& lower, const vector<T>& size)
{
         if(pos.x < lower.x) pos.x += size.x;
    else if(pos.x > upper.x) pos.x -= size.x;
         if(pos.y < lower.y) pos.y += size.y;
    else if(pos.y > upper.y) pos.y -= size.y;
         if(pos.z < lower.z) pos.z += size.z;
    else if(pos.z > upper.z) pos.z -= size.z;
    return pos;
}

} // lj
#endif// LJ_BOUNDARY_CONDITION
