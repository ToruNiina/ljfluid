#include "particle.hpp"
#include "boundary_condition.hpp"
#include <random>

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

    for(std::size_t timestep=0; timestep < 10000; ++timestep)
    {
        if(timestep % 16 == 0)
        {
            std::cout << ps << std::flush;
        }
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
    }
    std::cout << ps << std::flush;

    return 0;
}
