#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

namespace lj
{

template<typename realT>
struct vector
{
    realT x, y, z;
};

template<typename T>
inline vector<T> operator+(const vector<T>& lhs, const vector<T>& rhs) noexcept
{
    return vector<T>{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}
template<typename T>
inline vector<T> operator-(const vector<T>& lhs, const vector<T>& rhs) noexcept
{
    return vector<T>{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}
template<typename T>
inline vector<T> operator*(const vector<T>& lhs, const T rhs) noexcept
{
    return vector<T>{lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}
template<typename T>
inline vector<T> operator*(const T lhs, const vector<T>& rhs) noexcept
{
    return vector<T>{lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
}
template<typename T>
inline vector<T> operator/(const vector<T>& lhs, const T rhs) noexcept
{
    return vector<T>{lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
}

template<typename T>
inline T dot_product(const vector<T>& lhs, const vector<T>& rhs) noexcept
{
    return lhs.x * rhs.x + lhs.y * rhs.y, lhs.z * rhs.z;
}
template<typename T>
inline vector<T>
cross_product(const vector<T>& lhs, const vector<T>& rhs) noexcept
{
    return vector<T>{lhs.y * rhs.z - lhs.z * rhs.y,
                     lhs.z * rhs.x - lhs.x * rhs.z,
                     lhs.x * rhs.y - lhs.y * rhs.x};
}
template<typename T>
inline T length_sq(const vector<T>& lhs) noexcept
{
    return lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z;
}
template<typename T>
inline T length(const vector<T>& lhs) noexcept
{
    return std::sqrt(lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z);
}

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

int main()
{
    const lj::vector<double> upper{80.0, 80.0, 80.0};
    const lj::vector<double> lower{ 0.0,  0.0,  0.0};
    const lj::vector<double> L = upper - lower;
    const double sgm = 1.0;
    const double eps = 1.0;
    const double kB  = 1.986231313e-3;
    const double T   = 300.0;
    const double dt  = 0.1;

    std::vector<lj::particle<double>> ps(512/* = 8 * 8 * 8*/);
    {
        std::mt19937 mt(123456789);
        std::normal_distribution<double> boltz(0.0, std::sqrt(kB * T));
        for(std::size_t i=0; i<512; ++i)
        {
            ps[i].mass = 1.0;
            ps[i].position = lj::vector<double>{
               5.0 + 10.0 * ((i & 0b000000111) >> 0),
               5.0 + 10.0 * ((i & 0b000111000) >> 3),
               5.0 + 10.0 * ((i & 0b111000000) >> 6)};
            ps[i].velocity = lj::vector<double>{boltz(mt), boltz(mt), boltz(mt)};
            ps[i].force    = lj::vector<double>{0.0, 0.0, 0.0};
        }
    }

    std::cout << ps << std::flush;

    for(std::size_t timestep=0; timestep < 10000; ++timestep)
    {
        for(auto& p : ps)
        {
            p.position = lj::adjust_position(
                p.position + dt * p.velocity + (dt * dt / 2) * p.force / p.mass,
                upper, lower, L);
            p.velocity = p.velocity + (dt / 2) * p.force / p.mass;
        }
        // calc_force(ps)
        for(auto& p : ps)
        {
            p.velocity = p.velocity + (dt / 2) * p.force / p.mass;
            p.force    = lj::vector<double>{0.0, 0.0, 0.0};
        }
        if(timestep % 16 == 0)
        {
            std::cout << ps << std::flush;
        }
    }

    return 0;
}
