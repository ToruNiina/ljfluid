#ifndef LJ_BOUNDARY_CONDITION
#define LJ_BOUNDARY_CONDITION
#include <lj/vector.hpp>

namespace lj
{

template<typename Real>
struct periodic_boundary
{
    periodic_boundary(const vector<Real>& low, const vector<Real>& up) noexcept
        : width(up - low), halfw((up - low) / Real(2)), upper(up), lower(low)
    {}

    vector<Real> adjust_direction(vector<Real> pos) const noexcept
    {
        if     (pos.x < -halfw.x){pos.x += width.x;}
        else if(pos.x >= halfw.x){pos.x -= width.x;}
        if     (pos.y < -halfw.y){pos.y += width.y;}
        else if(pos.y >= halfw.y){pos.y -= width.y;}
        if     (pos.z < -halfw.z){pos.z += width.z;}
        else if(pos.z >= halfw.z){pos.z -= width.z;}
        return pos;
    }

    vector<Real> adjust_position(vector<Real> pos) const noexcept
    {
        if     (pos.x <  lower.x){pos.x += width.x;}
        else if(pos.x >= upper.x){pos.x -= width.x;}
        if     (pos.y <  lower.y){pos.y += width.y;}
        else if(pos.y >= upper.y){pos.y -= width.y;}
        if     (pos.z <  lower.z){pos.z += width.z;}
        else if(pos.z >= upper.z){pos.z -= width.z;}
        return pos;
    }

    vector<Real> width;
    vector<Real> halfw;
    vector<Real> upper;
    vector<Real> lower;
};

} // lj
#endif// LJ_BOUNDARY_CONDITION
