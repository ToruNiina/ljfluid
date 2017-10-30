#include <lj/particle.hpp>
#include <lj/boundary_condition.hpp>
#include <iterator>
#include <random>

namespace lj
{
constexpr static double sgm = 1.0;
constexpr static double eps = 1.0;

void calc_force(std::vector<particle<double>>& ps,
                const vector<double>& size, const vector<double>& size_half)
{
    for(auto iter(ps.begin()), iend(std::prev(ps.end())); iter != iend; ++iter)
    {
        const auto& pos1 = iter->position;
        for(auto jter(std::next(iter)), jend(ps.end()); jter != jend; ++jter)
        {
            const auto&  pos2 = jter->position;
            const auto   dpos = adjust_direction(pos2 - pos1, size, size_half);
            const double invr = 1. / length(dpos);
            const double sgmr = sgm * invr;
            const double f    = 24 * eps * (std::pow(sgmr, 6) - 2 * std::pow(sgmr, 12)) * invr;
            iter->force = iter->force + dpos * f;
            jter->force = jter->force - dpos * f;
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
double calc_potential_energy(const std::vector<particle<double>>& ps)
{
    double E = 0.0;
    for(auto iter(ps.begin()), iend(std::prev(ps.end())); iter != iend; ++iter)
    {
        const auto& pos1 = iter->position;
        for(auto jter(std::next(iter)), jend(ps.end()); jter != jend; ++jter)
        {
            const auto& pos2 = jter->position;
            const double sgmr = sgm / length(pos2 - pos1);
            E += 4 * eps * (std::pow(sgmr, 12) - std::pow(sgmr, 6));
        }
    }
    return E;
}
} // lj

int main()
{
    const lj::vector<double> upper{80.0, 80.0, 80.0};
    const lj::vector<double> lower{ 0.0,  0.0,  0.0};
    const lj::vector<double> L  = upper - lower;
    const lj::vector<double> Lh = L / 2.0;
    const double kB  = 1.986231313e-3;
    const double T   = 300.0;
    const double dt  = 0.01;

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
    for(std::size_t timestep=0; timestep < 10000; ++timestep)
    {
        if(timestep % 16 == 0)
        {
            const double Ek = lj::calc_kinetic_energy(ps);
            const double Ep = lj::calc_potential_energy(ps);
            std::cerr << timestep * dt << '\t' << Ek << '\t' << Ep << '\t'
                      << Ek + Ep << '\n';
            std::cout << ps << std::flush;
        }
        for(auto& p : ps)
        {
            p.position = lj::adjust_position(
                p.position + dt * p.velocity + (dt * dt / 2) * p.force / p.mass,
                upper, lower, L);
            p.velocity = p.velocity + (dt / 2) * p.force / p.mass;
        }
        lj::calc_force(ps, L, Lh);
        for(auto& p : ps)
        {
            p.velocity = p.velocity + (dt / 2) * p.force / p.mass;
            p.force    = lj::vector<double>{0.0, 0.0, 0.0};
        }
    }
    std::cout << ps << std::flush;

    return 0;
}
