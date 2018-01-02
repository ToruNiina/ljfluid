#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

namespace lj
{

template<typename T> constexpr T sgm = 1.0;
template<typename T> constexpr T eps = 1.0;

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
struct particle_container
{
    std::vector<realT> mass;
    std::vector<realT> rx, ry, rz;
    std::vector<realT> vx, vy, vz;
    std::vector<realT> fx, fy, fz;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const particle_container<T>& p)
{
    os << p.rx.size() << "\n\n";
    for(std::size_t i=0; i<p.rx.size(); ++i)
    {
        os << "C      "
           << std::fixed << std::setprecision(5) << std::showpoint
           << std::setw(10) << std::right << p.rx[i]
           << std::setw(10) << std::right << p.ry[i]
           << std::setw(10) << std::right << p.rz[i] << '\n';
    }
    return os;
}

template<typename T>
void calc_force(particle_container<T>& ps, const boundary<T>& bdy)
{
    for(std::size_t i=0, ie = ps.rx.size() - 1; i < ie; ++i)
    {
        const auto pos1 = vec<T>{ps.rx[i], ps.ry[i], ps.rz[i]};
        for(std::size_t j=i+1, je = ps.rx.size(); j < je; ++j)
        {
            const auto pos2 = vec<T>{ps.rx[j], ps.ry[j], ps.rz[j]};
            const auto dpos = adjust_direction(pos2 - pos1, bdy);
            const T    invr = 1. / length(dpos);
            const T    sgmr = sgm<T> * invr;
            const T    f    = 24 * eps<T> *
                (std::pow(sgmr, 6) - 2 * std::pow(sgmr, 12)) * invr;

            ps.fx[i] += dpos.x * f;
            ps.fy[i] += dpos.y * f;
            ps.fz[i] += dpos.z * f;

            ps.fx[j] -= dpos.x * f;
            ps.fy[j] -= dpos.y * f;
            ps.fz[j] -= dpos.z * f;
        }
    }
    return;
}

template<typename T>
T calc_kinetic_energy(const particle_container<T>& ps)
{
    T E = 0.0;
    for(std::size_t i=0; i<ps.vx.size(); ++i)
    {
        E += length_sq(vec<T>{ps.vx[i], ps.vy[i], ps.vz[i]}) * ps.mass[i] / 2;
    }
    return E;
}

template<typename T>
T calc_potential_energy(const particle_container<T>& ps, const boundary<T>& bdy)
{
    T E = 0.0;
    for(std::size_t i=0, ie = ps.rx.size() - 1; i < ie; ++i)
    {
        const auto pos1 = vec<T>{ps.rx[i], ps.ry[i], ps.rz[i]};

        for(std::size_t j=i+1, je = ps.rx.size(); j < je; ++j)
        {
            const auto pos2 = vec<T>{ps.rx[j], ps.ry[j], ps.rz[j]};
            const auto dpos = adjust_direction(pos2 - pos1, bdy);
            const T    sgmr = sgm<T> / length(dpos);
            E += 4 * eps<T> * (std::pow(sgmr, 12) - std::pow(sgmr, 6));
        }
    }
    return E;
}
} // lj

int main()
{
    std::ios_base::sync_with_stdio(false);
    using realT = double;

    const std::size_t num_p      = 4096;
    const std::size_t total_step = 10000;
    const realT dt  = 0.01;
    const realT kB  = 1.986231313e-3;
    const realT T   = 300.0;

    const auto bdy = lj::make_boundary<realT>({80.0, 80.0, 80.0});
    lj::particle_container<realT> ps;
    ps.mass.resize(num_p);
    ps.rx.resize(num_p);
    ps.ry.resize(num_p);
    ps.rz.resize(num_p);
    ps.vx.resize(num_p);
    ps.vy.resize(num_p);
    ps.vz.resize(num_p);
    ps.fx.resize(num_p);
    ps.fy.resize(num_p);
    ps.fz.resize(num_p);

    {
        std::mt19937 mt(123456789);
        std::normal_distribution<realT> boltz(0.0, std::sqrt(kB * T));
        for(std::size_t i=0; i<num_p; ++i)
        {
            ps.mass[i] = 1.0;
            ps.rx[i] = 2.5 + 5.0 * ((i & 0b000000001111) >> 0);
            ps.ry[i] = 2.5 + 5.0 * ((i & 0b000011110000) >> 4);
            ps.rz[i] = 2.5 + 5.0 * ((i & 0b111100000000) >> 8);
            ps.vx[i] = boltz(mt);
            ps.vy[i] = boltz(mt);
            ps.vz[i] = boltz(mt);
            ps.fx[i] = 0.0;
            ps.fy[i] = 0.0;
            ps.fz[i] = 0.0;
        }
    }

    std::cerr << "time\tkinetic\tpotential\ttotal\n";
    for(std::size_t timestep=0; timestep < total_step; ++timestep)
    {
        if(timestep % 16 == 0)
        {
            const realT Ek = lj::calc_kinetic_energy(ps);
            const realT Ep = lj::calc_potential_energy(ps, bdy);
            std::cerr << timestep * dt << '\t' << Ek << '\t' << Ep << '\t'
                      << Ek + Ep << '\n';
            std::cout << ps << std::flush;
        }

        for(std::size_t i=0; i<num_p; ++i)
        {
            ps.rx[i] += dt * ps.vx[i] + (dt * dt / 2) * ps.fx[i] / ps.mass[i];
            ps.ry[i] += dt * ps.vy[i] + (dt * dt / 2) * ps.fy[i] / ps.mass[i];
            ps.rz[i] += dt * ps.vz[i] + (dt * dt / 2) * ps.fz[i] / ps.mass[i];
                 if(ps.rx[i] < -bdy.size_half.x){ps.rx[i] += bdy.size.x;}
            else if(ps.rx[i] >  bdy.size_half.x){ps.rx[i] -= bdy.size.x;}
                 if(ps.ry[i] < -bdy.size_half.y){ps.ry[i] += bdy.size.y;}
            else if(ps.ry[i] >  bdy.size_half.y){ps.ry[i] -= bdy.size.y;}
                 if(ps.rz[i] < -bdy.size_half.z){ps.rz[i] += bdy.size.z;}
            else if(ps.rz[i] >  bdy.size_half.z){ps.rz[i] -= bdy.size.z;}

            ps.vx[i] += (dt / 2) * ps.fx[i] / ps.mass[i];
            ps.vy[i] += (dt / 2) * ps.fy[i] / ps.mass[i];
            ps.vz[i] += (dt / 2) * ps.fz[i] / ps.mass[i];
        }

        lj::calc_force(ps, bdy);

        for(std::size_t i=0; i<num_p; ++i)
        {
            ps.vx[i] += (dt / 2) * ps.fx[i] / ps.mass[i];
            ps.vy[i] += (dt / 2) * ps.fy[i] / ps.mass[i];
            ps.vz[i] += (dt / 2) * ps.fz[i] / ps.mass[i];

            ps.fx[i] = 0.0;
            ps.fy[i] = 0.0;
            ps.fz[i] = 0.0;
        }
    }
    std::cout << ps << std::flush;

    return 0;
}

