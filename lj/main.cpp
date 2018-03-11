#include <lj/particle.hpp>
#include <lj/boundary_condition.hpp>
// #include <lj/verletlist.hpp>
#include <lj/cell_list.hpp>
#include <iterator>
#include <random>

namespace lj
{
template<typename Real>
constexpr static Real sgm = 1.0;
template<typename Real>
constexpr static Real eps = 1.0;
template<typename Real>
constexpr static Real r_c = sgm<Real> * 2.5;
template<typename Real>
constexpr static Real inv_r_c = 1.0 / r_c<Real>;

template<typename Real>
void calc_force(std::vector<particle<Real>>& ps,
                const periodic_boundary<Real>& pb,
                const cell_list<Real>& ls)
//                 const verlet_list<Real>& ls)
{
    for(std::size_t i=0; i<ps.size(); ++i)
    {
        const auto& pos1 = ps[i].position;
//         for(std::size_t j=i+1; j<ps.size(); ++j)
        for(auto j : ls.neighbors(i))
        {
            const auto&  pos2 = ps[j].position;
            const auto   dpos = pb.adjust_direction(pos2 - pos1);
            const Real invr = 1. / length(dpos);
            if(invr < inv_r_c<Real>) {continue;}

            const Real sgmr  = sgm<Real> * invr;
            const Real sr6 = std::pow(sgmr, 6);
            const auto   f = dpos * (24 * eps<Real> * sr6 * (1 - 2 * sr6) * invr);
            ps[i].force += f;
            ps[j].force -= f;
        }
    }
    return;
}

template<typename Real>
Real calc_kinetic_energy(const std::vector<particle<Real>>& ps)
{
    Real E = 0.0;
    for(const auto& p : ps)
    {
        E += length_sq(p.velocity) * p.mass / 2;
    }
    return E;
}

template<typename Real>
Real calc_potential_energy(const std::vector<particle<Real>>& ps,
                             const periodic_boundary<Real>& pb,
                             const cell_list<Real>& ls)
//                              const verlet_list<Real>& ls)
{
    Real E = 0.0;
    for(std::size_t i=0; i<ps.size(); ++i)
    {
        const auto& pos1 = ps[i].position;
//         for(std::size_t j=i+1; j<ps.size(); ++j)
        for(auto j : ls.neighbors(i))
        {
            const auto& pos2 = ps[j].position;
            const Real invr = 1.0 / length(pb.adjust_direction(pos2 - pos1));
            if(invr < inv_r_c<Real>) {continue;}

            const Real sgmr = sgm<Real> * invr;
            const Real sr6 = std::pow(sgmr, 6);
            E += 4 * eps<Real> * sr6 * (sr6 - 1);
        }
    }
    return E;
}
} // lj

int main()
{
//     typedef double Real;
    typedef float Real;
    const std::size_t log2N = 4;

    const std::size_t N = std::pow(2, log2N);
    const lj::vector<Real> upper{N*2.0, N*2.0, N*2.0};
    const lj::vector<Real> lower{  0.0,   0.0,   0.0};
    const lj::periodic_boundary<Real> pb(lower, upper);

    const Real kB  = 1.986231313e-3;
    const Real T   = 300.0;
    const Real dt  = 0.01;

//     lj::verlet_list<Real> ls(dt, lj::r_c<Real>, 0.25);
    lj::cell_list<Real> ls(dt, lj::r_c<Real>, 0.1, pb);

    std::vector<lj::particle<Real>> ps(N * N * N);
    {
        std::mt19937 mt(123456789);
        std::normal_distribution<Real> boltz(0.0, std::sqrt(kB * T));
        for(std::size_t i=0; i<ps.size(); ++i)
        {
            ps[i].mass = 1.0;
            ps[i].position = lj::vector<Real>{
               Real(1) + Real(2) * ((i & (N-1) << log2N * 0) >> log2N * 0),
               Real(1) + Real(2) * ((i & (N-1) << log2N * 1) >> log2N * 1),
               Real(1) + Real(2) * ((i & (N-1) << log2N * 2) >> log2N * 2)};
            ps[i].velocity = lj::vector<Real>{boltz(mt), boltz(mt), boltz(mt)};
            ps[i].force    = lj::vector<Real>{0.0, 0.0, 0.0};
        }
    }

    ls.make(ps, pb);

    std::cerr << "time\tkinetic\tpotential\ttotal\n";
    lj::calc_force(ps, pb, ls);
    for(std::size_t timestep=0; timestep < 100000; ++timestep)
    {
        if(timestep % 1000 == 0)
        {
            const Real Ek = lj::calc_kinetic_energy(ps);
            const Real Ep = lj::calc_potential_energy(ps, pb, ls);
            std::cerr << timestep * dt << '\t' << Ek << '\t' << Ep << '\t'
                      << Ek + Ep << '\n';
            std::cout << ps << std::flush;
        }

        Real max_vel2 = 0.0;
        for(auto& p : ps)
        {
            max_vel2 = std::max(max_vel2, length_sq(p.velocity));
            p.position = pb.adjust_position(p.position + dt * p.velocity +
                                            (dt * dt / 2) * p.force / p.mass);
            p.velocity = p.velocity + (dt / 2) * p.force / p.mass;
        }

        ls.update(ps, pb, std::sqrt(max_vel2));
        lj::calc_force(ps, pb, ls);

        for(auto& p : ps)
        {
            p.velocity = p.velocity + (dt / 2) * p.force / p.mass;
            p.force    = lj::vector<Real>{0.0, 0.0, 0.0};
        }
    }
    std::cout << ps << std::flush;

    return 0;
}
