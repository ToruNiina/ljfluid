#include <lj/particle.hpp>
#include <lj/boundary_condition.hpp>
#include <lj/verletlist.hpp>
#include <iterator>
#include <random>

namespace lj
{
constexpr static double sgm = 1.0;
constexpr static double eps = 1.0;
constexpr static double r_c = sgm * 2.5;
constexpr static double inv_r_c = 1.0 / r_c;

void calc_force(std::vector<particle<double>>& ps,
                const periodic_boundary<double>& pb,
                const verlet_list<double>& vl)
{
    for(std::size_t i=0; i<ps.size(); ++i)
    {
        const auto& pos1 = ps[i].position;
        for(auto j : vl.neighbors(i))
        {
            const auto&  pos2 = ps[j].position;
            const auto   dpos = pb.adjust_direction(pos2 - pos1);
            const double invr = 1. / length(dpos);
            if(invr < inv_r_c) {continue;}

            const double sgmr  = sgm * invr;
            const double sr6 = std::pow(sgmr, 6);
            const auto   f = dpos * (24 * eps * sr6 * (1.0 - 2 * sr6) * invr);
            ps[i].force += f;
            ps[j].force -= f;
        }
    }
    return;
}

double calc_kinetic_energy(const std::vector<particle<double>>& ps)
{
    double E = 0.0;
    for(const auto& p : ps)
    {
        E += length_sq(p.velocity) * p.mass / 2;
    }
    return E;
}

double calc_potential_energy(const std::vector<particle<double>>& ps,
                             const periodic_boundary<double>& pb,
                             const verlet_list<double>& vl)
{
    double E = 0.0;
    for(std::size_t i=0; i<ps.size(); ++i)
    {
        const auto& pos1 = ps[i].position;
        for(auto j : vl.neighbors(i))
        {
            const auto& pos2 = ps[j].position;
            const double invr = 1.0 / length(pb.adjust_direction(pos2 - pos1));
            if(invr < inv_r_c) {continue;}

            const double sgmr = sgm * invr;
            const double sr6 = std::pow(sgmr, 6);
            E += 4 * eps * sr6 * (sr6 - 1.0);
        }
    }
    return E;
}
} // lj

int main()
{
    const lj::vector<double> upper{80.0, 80.0, 80.0};
    const lj::vector<double> lower{ 0.0,  0.0,  0.0};
    const lj::periodic_boundary<double> pb(lower, upper);

    const double kB  = 1.986231313e-3;
    const double T   = 300.0;
    const double dt  = 0.01;

    lj::verlet_list<double> vl(dt, lj::r_c, 0.25);

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

    std::cerr << "time\tkinetic\tpotential\ttotal\n";
    vl.make(ps, pb);
    lj::calc_force(ps, pb, vl);
    for(std::size_t timestep=0; timestep < 10000; ++timestep)
    {
        if(timestep % 16 == 0)
        {
            const double Ek = lj::calc_kinetic_energy(ps);
            const double Ep = lj::calc_potential_energy(ps, pb, vl);
            std::cerr << timestep * dt << '\t' << Ek << '\t' << Ep << '\t'
                      << Ek + Ep << '\n';
            std::cout << ps << std::flush;
        }

        double max_vel2 = 0.0;
        for(auto& p : ps)
        {
            max_vel2 = std::max(max_vel2, length_sq(p.velocity));
            p.position = pb.adjust_position(p.position + dt * p.velocity +
                                            (dt * dt / 2) * p.force / p.mass);
            p.velocity = p.velocity + (dt / 2) * p.force / p.mass;
        }

        vl.update(ps, pb, std::sqrt(max_vel2));
        lj::calc_force(ps, pb, vl);

        for(auto& p : ps)
        {
            p.velocity = p.velocity + (dt / 2) * p.force / p.mass;
            p.force    = lj::vector<double>{0.0, 0.0, 0.0};
        }
    }
    std::cout << ps << std::flush;

    return 0;
}
