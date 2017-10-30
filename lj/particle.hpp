#ifndef LJ_PARTICLE
#define LJ_PARTICLE
#include <lj/vector.hpp>
#include <vector>
#include <iostream>
#include <iomanip>

namespace lj
{

template<typename realT>
struct particle
{
    realT         mass;
    vector<realT> position;
    vector<realT> velocity;
    vector<realT> force;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const particle<T>& p)
{
    os << "C      " << std::fixed << std::setprecision(5) << std::showpoint
       << std::setw(10) << std::right << p.position.x
       << std::setw(10) << std::right << p.position.y
       << std::setw(10) << std::right << p.position.z;
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<particle<T>>& ps)
{
    os << ps.size() << '\n';
    os << '\n';
    for(const auto& p : ps)
    {
        os << p << '\n';
    }
    return os;
}

} // lj
#endif
