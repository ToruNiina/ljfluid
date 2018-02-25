#ifndef LJ_PARTICLE
#define LJ_PARTICLE
#include <cuda/vec.cuh>
#include <thrust/host_vector.h>
#include <vector>
#include <iostream>
#include <iomanip>

namespace lj
{

struct particle
{
    float  mass;
    float4 position;
    float4 velocity;
    float4 force;
};

__host__
inline std::ostream&
operator<<(std::ostream& os, const particle& p)
{
    os << "H      " << std::fixed << std::setprecision(4) << std::showpoint
       << std::setw(10) << std::right << p.position.x
       << std::setw(10) << std::right << p.position.y
       << std::setw(10) << std::right << p.position.z;
    return os;
}

__host__
inline std::ostream&
operator<<(std::ostream& os, const std::vector<particle>& ps)
{
    os << ps.size() << "\n\n";
    for(const auto& p : ps)
    {
        os << p << '\n';
    }
    return os;
}

__host__
inline std::ostream&
operator<<(std::ostream& os, const thrust::host_vector<particle>& ps)
{
    os << ps.size() << "\n\n";
    for(const auto& p : ps)
    {
        os << p << '\n';
    }
    return os;
}

template<typename Alloc>
__host__
inline std::ostream&
operator<<(std::ostream& os, const thrust::device_vector<particle, Alloc>& ps)
{
    const thrust::host_vector<particle> tmp = ps;
    os << tmp;
    return os;
}

} // lj
#endif
