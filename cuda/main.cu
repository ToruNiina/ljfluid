#include <cuda/particle.cuh>
#include <cuda/boundary_condition.cuh>
#include <cuda/grid.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <curand.h>
#include <iterator>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

namespace lj
{
__device__ __host__ constexpr float sgm() noexcept {return 1.0f;}
__device__ __host__ constexpr float eps() noexcept {return 1.0f;}
__device__ __host__ constexpr float cutoff() noexcept {return 2.5f;}
__device__ __host__ constexpr float mergin() noexcept {return 0.1f;}

struct kinetic_energy_calculator
    : thrust::binary_function<float, float4, float>
{
    __device__ __host__
    float operator()(float m, float4 v) const noexcept
    {
        return m * length_sq(v);
    };
};

float kinetic_energy(const particle_container& ps)
{
    return thrust::inner_product(
            ps.device_masses.cbegin(),      ps.device_masses.cend(),
            ps.device_velocities.cbegin(), 0.0, thrust::plus<float>(),
            kinetic_energy_calculator()) * 0.5;
}

struct velocity_verlet_update_1
{
    velocity_verlet_update_1(float dt_, periodic_boundary b_)
        : dt(dt_), dt_half(dt_ * 0.5), b(b_)
    {}

    template<typename Tuple>
    __device__ __host__
    void operator()(Tuple mpvf) const noexcept
    {
        thrust::get<1>(mpvf) = adjust_position(
                thrust::get<1>(mpvf) + dt * thrust::get<2>(mpvf) +
                (dt * dt_half / thrust::get<0>(mpvf)) * thrust::get<3>(mpvf),
                b);

        thrust::get<2>(mpvf) = thrust::get<2>(mpvf) +
                (dt_half / thrust::get<0>(mpvf)) * thrust::get<3>(mpvf);
        return;
    }

    const float dt;
    const float dt_half;
    const periodic_boundary b;
};

struct velocity_verlet_update_2
{
    velocity_verlet_update_2(float dt_)
        : dt(dt_), dt_half(dt_ * 0.5)
    {}

    template<typename Tuple>
    __device__ __host__
    void operator()(Tuple mvf) const noexcept
    {
        thrust::get<1>(mvf) = thrust::get<1>(mvf) +
                (dt_half / thrust::get<0>(mvf)) * thrust::get<2>(mvf);
        thrust::get<2>(mvf) = make_float4(0,0,0,0);
        return;
    }

    const float dt;
    const float dt_half;
};

} // lj

struct tuple_vector_converter
{
    __device__ __host__
    float4 operator()(const thrust::tuple<float, float, float>& t) const noexcept
    {
        return make_float4(
                thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t), 0.0);
    };
};

struct position_initializer
{
    __device__ __host__
    float4 operator()(std::size_t i) const noexcept
    {
        return make_float4(1.0 + 2.0 * ((i & 0b000011) >> 0),
                           1.0 + 2.0 * ((i & 0b001100) >> 2),
                           1.0 + 2.0 * ((i & 0b110000) >> 4),
                           0.0);
    };
};

int main()
{
    const float4 upper    = make_float4( 8.0,  8.0,  8.0, 0.0);
    const float4 lower    = make_float4( 0.0,  0.0,  0.0, 0.0);
    const auto   boundary = lj::make_boundary(lower, upper);

    const std::size_t step = 100000;
    const std::size_t N    = std::pow(4, 3);
    const std::size_t seed = 123456789;
    const float kB  = 1.986231313e-3;
    const float T   = 300.0;
    const float dt  = 0.01;

    lj::particle_container ps(N);

    /* initialization */{
        thrust::fill(ps.device_masses.begin(), ps.device_masses.end(), 1.0f);
        thrust::fill(ps.host_masses.begin(),   ps.host_masses.end(),   1.0f);

        thrust::transform(
            /* input  begin */ thrust::make_counting_iterator<std::size_t>(0),
            /* input  end   */ thrust::make_counting_iterator<std::size_t>(N),
            /* output begin */ ps.device_positions.begin(),
            /* conversion   */ position_initializer());

        // prepair cuRAND generators
        curandGenerator_t rng;
        const auto st_gen = curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
        assert(st_gen  == CURAND_STATUS_SUCCESS);
        const auto st_seed = curandSetPseudoRandomGeneratorSeed(rng, seed);
        assert(st_seed == CURAND_STATUS_SUCCESS);

        thrust::device_vector<float> boltz_x(N);
        thrust::device_vector<float> boltz_y(N);
        thrust::device_vector<float> boltz_z(N);
        {
            const auto st_genrnd = curandGenerateNormal(
                    rng, boltz_x.data().get(), N, 0.0, std::sqrt(kB * T));
            assert(st_genrnd == CURAND_STATUS_SUCCESS);
        }

        {
            const auto st_genrnd = curandGenerateNormal(
                    rng, boltz_y.data().get(), N, 0.0, std::sqrt(kB * T));
            assert(st_genrnd == CURAND_STATUS_SUCCESS);
        }

        {
            const auto st_genrnd = curandGenerateNormal(
                    rng, boltz_z.data().get(), N, 0.0, std::sqrt(kB * T));
            assert(st_genrnd == CURAND_STATUS_SUCCESS);
        }

        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(
                boltz_x.begin(), boltz_y.begin(), boltz_z.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                boltz_x.end(),   boltz_y.end(),   boltz_z.end())),
            ps.device_velocities.begin(), tuple_vector_converter());
    }

    ps.pull_device_particles();

    {
        std::ofstream traj("traj.xyz");
        traj << ps.host_positions.size() << "\n\n";
        for(auto iter = ps.host_positions.begin(), iend = ps.host_positions.end();
                iter != iend; ++iter)
        {
            const auto& v = *iter;
            traj << "H      " << std::fixed << std::setprecision(5) << std::showpoint
                 << std::setw(10) << std::right << v.x
                 << std::setw(10) << std::right << v.y
                 << std::setw(10) << std::right << v.z << '\n';
        }
    }

    std::cerr << "kinetic energy = " << lj::kinetic_energy(ps)
              << ", 3/2 NkBT = " << N * kB * T * 1.5 << std::endl;

    //TODO: add potential
    lj::grid grid(lj::sgm() * lj::cutoff(), lj::mergin(), boundary);
//     std::cerr << grid.Nx << std::endl;
//     std::cerr << grid.Ny << std::endl;
//     std::cerr << grid.Nz << std::endl;

    grid.assign(ps.device_positions);

    std::cerr << "assigned" << std::endl;

    for(std::size_t i=0; i<ps.device_positions.size(); ++i)
    {
        const std::size_t offset = i * grid.stride;
        std::cerr << "num_neighbors = " << grid.number_of_neighbors[i] << std::endl;
        std::cerr << '{';
        for(std::size_t n=0; n<grid.stride; ++n)
        {
            std::cerr << grid.verlet_list[n + offset] << ", ";
        }
        std::cerr << "}\n";
    }
    std::cerr << std::endl;

    const lj::velocity_verlet_update_1 update1(dt, boundary);
    const lj::velocity_verlet_update_2 update2(dt);
    for(std::size_t s=0; s < step; ++s)
    {
        if(s % 1000 == 0)
        {
            ps.pull_device_particles();

            std::ofstream traj("traj.xyz",
                    std::ios_base::app | std::ios_base::out);

            traj << ps.host_positions.size() << "\n\n";
            for(auto iter = ps.host_positions.begin(), iend = ps.host_positions.end();
                    iter != iend; ++iter)
            {
                const auto& v = *iter;
                traj << "H      " << std::fixed << std::setprecision(5) << std::showpoint
                     << std::setw(10) << std::right << v.x
                     << std::setw(10) << std::right << v.y
                     << std::setw(10) << std::right << v.z << '\n';
            }
        }

        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                ps.device_masses.begin(), ps.device_positions.begin(),
                ps.device_velocities.begin(), ps.device_forces.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                ps.device_masses.end(), ps.device_positions.end(),
                ps.device_velocities.end(), ps.device_forces.end())),
            update1);

        // calc force here

        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                ps.device_masses.begin(),
                ps.device_velocities.begin(), ps.device_forces.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                ps.device_masses.end(),
                ps.device_velocities.end(), ps.device_forces.end())),
            update2);
    }

    return 0;
}
