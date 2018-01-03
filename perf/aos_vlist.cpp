#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <random>
#include <cmath>

namespace lj
{

template<typename T> constexpr T sgm = 1.0;
template<typename T> constexpr T eps = 1.0;
template<typename T> constexpr T r_c = 2.5 * sgm<T>;
template<typename T> constexpr T mgn = 1.2; // safety mergin

template<typename realT>
struct vec
{
    realT x, y, z;
};

template<typename T>
inline vec<T> operator+(const vec<T>& lhs, const vec<T>& rhs) noexcept
{
    return vec<T>{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}
template<typename T>
inline vec<T> operator-(const vec<T>& lhs, const vec<T>& rhs) noexcept
{
    return vec<T>{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}
template<typename T>
inline vec<T> operator*(const vec<T>& lhs, const T rhs) noexcept
{
    return vec<T>{lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}
template<typename T>
inline vec<T> operator*(const T lhs, const vec<T>& rhs) noexcept
{
    return vec<T>{lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
}
template<typename T>
inline vec<T> operator/(const vec<T>& lhs, const T rhs) noexcept
{
    return vec<T>{lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
}

template<typename T>
inline T dot_product(const vec<T>& lhs, const vec<T>& rhs) noexcept
{
    return lhs.x * rhs.x + lhs.y * rhs.y, lhs.z * rhs.z;
}
template<typename T>
inline vec<T>
cross_product(const vec<T>& lhs, const vec<T>& rhs) noexcept
{
    return vec<T>{lhs.y * rhs.z - lhs.z * rhs.y,
                  lhs.z * rhs.x - lhs.x * rhs.z,
                  lhs.x * rhs.y - lhs.y * rhs.x};
}
template<typename T>
inline T length_sq(const vec<T>& lhs) noexcept
{
    return lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z;
}
template<typename T>
inline T length(const vec<T>& lhs) noexcept
{
    return std::sqrt(length_sq(lhs));
}

template<typename T>
struct boundary
{
    vec<T> upper;
    vec<T> lower;
    vec<T> size;
    vec<T> size_half;
};

template<typename T>
inline boundary<T> make_boundary(const vec<T>& upper) noexcept
{
    return boundary<T>{upper, {0, 0, 0}, upper, upper * 0.5};
}

template<typename T>
vec<T> adjust_direction(vec<T> pos, const boundary<T>& bdy) noexcept
{
         if(pos.x < -bdy.size_half.x){pos.x += bdy.size.x;}
    else if(pos.x >  bdy.size_half.x){pos.x -= bdy.size.x;}
         if(pos.y < -bdy.size_half.y){pos.y += bdy.size.y;}
    else if(pos.y >  bdy.size_half.y){pos.y -= bdy.size.y;}
         if(pos.z < -bdy.size_half.z){pos.z += bdy.size.z;}
    else if(pos.z >  bdy.size_half.z){pos.z -= bdy.size.z;}
    return pos;
}

template<typename T>
vec<T> adjust_position(vec<T> pos, const boundary<T>& bdy) noexcept
{
         if(pos.x < bdy.lower.x) {pos.x += bdy.size.x;}
    else if(pos.x > bdy.upper.x) {pos.x -= bdy.size.x;}
         if(pos.y < bdy.lower.y) {pos.y += bdy.size.y;}
    else if(pos.y > bdy.upper.y) {pos.y -= bdy.size.y;}
         if(pos.z < bdy.lower.z) {pos.z += bdy.size.z;}
    else if(pos.z > bdy.upper.z) {pos.z -= bdy.size.z;}
    return pos;
}

template<typename realT>
struct particle
{
    realT      mass;
    vec<realT> position;
    vec<realT> velocity;
    vec<realT> force;
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
    os << ps.size() << "\n\n";
    for(const auto& p : ps) {os << p << '\n';}
    return os;
}

template<typename T>
T make_verlet_list(const std::vector<particle<T>>& ps, const boundary<T>& bdy,
                   std::vector<std::vector<std::size_t>>& vls)
{
    vls.resize(ps.size());
    for(auto& ls : vls){ls.clear();}

    constexpr T threshold = r_c<T> * mgn<T>;
    constexpr T mergin    = threshold - r_c<T>;
    constexpr T thr2      = threshold * threshold;

    for(std::size_t i=0, ie=ps.size()-1; i<ie; ++i)
    {
        const auto& pos1 = ps[i].position;
        for(std::size_t j=i+1, je=ps.size(); j<je; ++j)
        {
            const auto& pos2 = ps[j].position;
            const auto  r2 = length_sq(adjust_direction(pos2 - pos1, bdy));
            if(r2 < thr2)
            {
                vls[i].push_back(j);
            }
        }
    }
    return mergin;
}

template<typename T>
void calc_force(std::vector<particle<T>>& ps, const boundary<T>& bdy,
                const std::vector<std::vector<std::size_t>>& vls)
{
    constexpr T r_c2 = r_c<T> * r_c<T>;
    std::size_t idx = 0;
    for(const auto& ls : vls)
    {
        const auto& pos1 = ps[idx].position;
        for(const auto jdx : ls)
        {
            const auto& pos2 = ps[jdx].position;
            const auto  dpos = adjust_direction(pos2 - pos1, bdy);
            const T     r2   = length_sq(dpos);
            if(r2 > r_c2){continue;}
            const T     invr = 1. / std::sqrt(r2);
            const T     sgmr = sgm<T> * invr;
            const T     f    = 24 * eps<T> * (std::pow(sgmr, 6) - 2 * std::pow(sgmr, 12)) * invr;
            ps[idx].force = ps[idx].force + dpos * f;
            ps[jdx].force = ps[jdx].force - dpos * f;
        }
        ++idx;
    }
    return;
}

template<typename T>
T calc_kinetic_energy(const std::vector<particle<T>>& ps)
{
    T E = 0.0;
    for(const auto& p : ps)
    {
        E += length_sq(p.velocity) * p.mass / 2;
    }
    return E;
}

template<typename T>
T calc_potential_energy(const std::vector<particle<T>>& ps, const boundary<T>& bdy,
                const std::vector<std::vector<std::size_t>>& vls)
{
    T E = 0.0;
    constexpr T r_c2 = r_c<T> * r_c<T>;
    std::size_t idx = 0;
    for(const auto& ls : vls)
    {
        const auto& pos1 = ps[idx].position;
        for(const auto jdx : ls)
        {
            const auto& pos2 = ps[jdx].position;
            const auto  dpos = adjust_direction(pos2 - pos1, bdy);
            const T     r2   = length_sq(dpos);
            if(r2 > r_c2){continue;}
            const T     sgmr = sgm<T> / std::sqrt(r2);
            E += 4 * eps<T> * (std::pow(sgmr, 12) - std::pow(sgmr, 6));
        }
        ++idx;
    }
    return E;
}

} // lj

int main()
{
    std::ios_base::sync_with_stdio(false);
    using realT = double;

    const std::size_t num_p      = 262144; /* = 64^3*/
    const std::size_t total_step = 10000;
    const realT dt  = 0.01;
    const realT kB  = 1.986231313e-3;
    const realT T   = 300.0;

    const auto bdy = lj::make_boundary<realT>({128.0, 128.0, 128.0});
    std::vector<lj::particle<realT>> ps(num_p);

    {
        std::mt19937 mt(123456789);
        std::normal_distribution<realT> boltz(0.0, std::sqrt(kB * T));
        for(std::size_t i=0; i<num_p; ++i)
        {
            ps[i].mass = 1.0;
            ps[i].position = lj::vec<realT>{
               0.5 + 2.0 * ((i & 0b000000000000111111) >> 0),
               0.5 + 2.0 * ((i & 0b000000111111000000) >> 6),
               0.5 + 2.0 * ((i & 0b111111000000000000) >> 12)};
            ps[i].velocity = lj::vec<realT>{boltz(mt), boltz(mt), boltz(mt)};
            ps[i].force    = lj::vec<realT>{0.0, 0.0, 0.0};
        }
    }

    std::vector<std::vector<std::size_t>> verlist(num_p);
    realT mergin = make_verlet_list(ps, bdy, verlist);

    std::cerr << "time\tkinetic\tpotential\ttotal\n";
    for(std::size_t timestep=0; timestep < total_step; ++timestep)
    {
        if(timestep % 16 == 0)
        {
            const realT Ek = lj::calc_kinetic_energy(ps);
            const realT Ep = lj::calc_potential_energy(ps, bdy, verlist);
            std::cerr << timestep * dt << '\t' << Ek << '\t' << Ep << '\t'
                      << Ek + Ep << '\n';
            std::cout << ps << std::flush;
        }

        for(auto& p : ps)
        {
            p.position = lj::adjust_position(
                p.position + dt * p.velocity + (dt * dt / 2) * p.force / p.mass,
                bdy);
            p.velocity = p.velocity + (dt / 2) * p.force / p.mass;
        }

        lj::calc_force(ps, bdy, verlist);

        realT max_vel2 = 0.0;
        for(auto& p : ps)
        {
            p.velocity = p.velocity + (dt / 2) * p.force / p.mass;
            p.force    = lj::vec<realT>{0.0, 0.0, 0.0};

            const auto vel2 = length_sq(p.velocity);
            if(max_vel2 < vel2)
            {
                max_vel2 = vel2;
            }
        }

        mergin -= std::sqrt(max_vel2) * dt;
        if(mergin < 0.0)
        {
            mergin = make_verlet_list(ps, bdy, verlist);
        }
    }
    std::cout << ps << std::flush;

    return 0;
}

