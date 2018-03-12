#include <cuda/particle.cuh>
#include <cuda/boundary_condition.cuh>
#include <cuda/grid.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

namespace lj
{
__device__ __host__ constexpr float sgm()    noexcept {return 1.0f;}
__device__ __host__ constexpr float eps()    noexcept {return 1.0f;}
__device__ __host__ constexpr float cutoff() noexcept {return 2.5f;}
__device__ __host__ constexpr float mergin() noexcept {return 0.1f;}

struct force_calculator
{
    __host__
    force_calculator(const float4* const ps, const std::size_t* const vl,
                     const std::size_t* nneigh, const std::size_t stride_,
                     const periodic_boundary b)
        : positions(ps), verlet_list(vl), num_neighbors(nneigh),
          inv_r_c(1.0 / (sgm() * cutoff())), stride(stride_), boundary(b)
    {}

    __host__ __device__
    float4 operator()(const std::size_t i) const noexcept
    {
        float4 force = make_float4(0.0, 0.0, 0.0, 0.0);
        const float4 pos1 = *(positions + i);

        const std::size_t n_neigh = *(num_neighbors + i);
        const std::size_t offset  = i * stride;

        for(std::size_t idx=0; idx<n_neigh; ++idx)
        {
            assert(idx != std::numeric_limits<std::size_t>::max());

            const std::size_t j = *(verlet_list + offset + idx);

            const float4 pos2 = *(positions + j);
            const float4 dpos = adjust_direction(pos2 - pos1, boundary);
            const float  invr = rsqrtf(length_sq(dpos));
            if(invr < inv_r_c) {continue;}

            const float sgmr = sgm() * invr;
            const float sr3  = sgmr * sgmr * sgmr;
            const float sr6  = sr3 * sr3;
            const float coef = 24 * eps() * sr6 * (1.0 - 2 * sr6) * invr;

            force = force + dpos * coef;
        }
        return force;
    }

    const float4*      const positions;
    const std::size_t* const verlet_list;
    const std::size_t* const num_neighbors;
    const float inv_r_c;
    const std::size_t stride;
    const periodic_boundary  boundary;
};

struct energy_calculator
{
    __host__
    energy_calculator(const float4* const ps, const std::size_t* const vl,
                      const std::size_t* nneigh, const std::size_t stride_,
                      const periodic_boundary b)
        : positions(ps), verlet_list(vl), num_neighbors(nneigh),
          inv_r_c(1.0 / (sgm() * cutoff())), stride(stride_), boundary(b)
    {}

    __device__ __host__
    float operator()(const std::size_t i) const noexcept
    {
        float energy = 0.0;

        const float4      pos1    = positions[i];
        const std::size_t n_neigh = num_neighbors[i];
        const std::size_t offset  = i * stride;

        for(std::size_t idx=0; idx<n_neigh; ++idx)
        {
            assert(idx != std::numeric_limits<std::size_t>::max());
            const std::size_t j = verlet_list[offset + idx];
            assert(j < 64);
            const float4 pos2 = positions[j];

            const float4 dpos = adjust_direction(pos2 - pos1, boundary);
            const float  invr = rsqrtf(length_sq(dpos));
            if(invr < inv_r_c) {continue;}

            const float  sgmr = sgm() * invr;
            const float  sr3  = sgmr * sgmr * sgmr;
            const float  sr6  = sr3 * sr3;
            energy += 4 * eps() * sr6 * (sr6 - 1.0);
        }

        return energy / 2;
    }

    const float4*      const positions;
    const std::size_t* const verlet_list;
    const std::size_t* const num_neighbors;
    const float inv_r_c;
    const std::size_t stride;
    const periodic_boundary boundary;
};

struct kinetic_energy_calculator
{
    __device__ __host__
    float operator()(float m, float4 v) const noexcept
    {
        return m * length_sq(v);
    }
};

float calc_kinetic_energy(const particle_container& ps)
{
    return thrust::inner_product(
            ps.device_masses.cbegin(),     ps.device_masses.cend(),
            ps.device_velocities.cbegin(), 0.0,
            thrust::plus<float>(), kinetic_energy_calculator()) * 0.5;
}

struct velocity_size_comparator
{
    __device__ __host__
    bool operator()(float4 lhs, float4 rhs) const noexcept
    {
        return length_sq(lhs) < length_sq(rhs);
    }
};

struct velocity_verlet_update_1
{
    velocity_verlet_update_1(float dt_, periodic_boundary b_)
        : dt(dt_), dt_half(dt_ / 2), b(b_)
    {}

    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple mpvf) const noexcept
    {
        float const   m = thrust::get<0>(mpvf);
        float4&       p = thrust::get<1>(mpvf);
        float4&       v = thrust::get<2>(mpvf);
        float4&       f = thrust::get<3>(mpvf);

        v += (dt / 2) * f / m;
        p = adjust_position(p + v * dt, b);
        f = make_float4(0,0,0,0);
        return;
    }

    const float dt;
    const float dt_half;
    const periodic_boundary b;
};


struct velocity_verlet_update_2
{
    velocity_verlet_update_2(float dt_)
        : dt(dt_), dt_half(dt_ / 2)
    {}

    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple mvf) const noexcept
    {
        float  const  m = thrust::get<0>(mvf);
        float4&       v = thrust::get<1>(mvf);
        float4 const& f = thrust::get<2>(mvf);

        v += (dt / 2) * f / m;
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
    position_initializer(std::size_t n)
        : N(n), bitflag(std::pow(2, n) - 1)
    {}

    __device__ __host__
    float4 operator()(std::size_t i) const noexcept
    {
        return make_float4(1.0 + 2.0 * ((i & (bitflag)       )       ),
                           1.0 + 2.0 * ((i & (bitflag) << N*1) >> N*1),
                           1.0 + 2.0 * ((i & (bitflag) << N*2) >> N*2),
                           0.0);
    };

    const std::size_t N, bitflag;
};

int main()
{
    const std::size_t log2NperEdge = 2;
    const std::size_t NperEdge     = std::pow(2, log2NperEdge);

    const std::size_t N    = std::pow(NperEdge, 3);
    const std::size_t step = 1000000;
    const float kB  = 1.986231313e-3;
    const float T   = 300.0;
    const float dt  = 0.01;

    const float  width    = 2.0 * NperEdge;
    const float4 upper    = make_float4(width, width, width, 0.0);
    const float4 lower    = make_float4(0.0,   0.0,   0.0,   0.0);
    const auto   boundary = lj::make_boundary(lower, upper);

    lj::particle_container ps(N);

    /* initialization */{
        thrust::fill(ps.device_masses.begin(), ps.device_masses.end(), 1.0f);
        thrust::transform(
            thrust::make_counting_iterator<std::size_t>(0),
            thrust::make_counting_iterator<std::size_t>(N),
            ps.device_positions.begin(),
            position_initializer(log2NperEdge));

        std::mt19937 mt(123456789);
        std::normal_distribution<float> boltz(0.0f, std::sqrt(kB * T));
        for(std::size_t i=0; i<N; ++i)
        {
            const float vx = boltz(mt);
            const float vy = boltz(mt);
            const float vz = boltz(mt);
            ps.host_velocities[i] = make_float4(vx, vy, vz, 0.0);
        }
        ps.device_velocities = ps.host_velocities;

        thrust::fill(ps.device_forces.begin(), ps.device_forces.end(),
                     make_float4(0, 0, 0, 0));
    }
    cudaDeviceSynchronize();

    {
        std::ofstream traj("traj.xyz");
        std::ofstream velo("velo.xyz");
    }

    lj::grid grid(lj::sgm() * lj::cutoff(), lj::mergin(), boundary);
    grid.assign(ps.device_positions);

//    std::cerr << "assigned" << std::endl;
//    for(std::size_t i=0; i<ps.device_positions.size(); ++i)
//    {
//        const std::size_t offset = i * grid.stride;
//        std::cerr << "num_neighbors = " << grid.number_of_neighbors[i] << std::endl;
//        std::cerr << '{';
//        for(std::size_t n=0; n<grid.stride; ++n)
//        {
//            std::cerr << grid.verlet_list[n + offset] << ", ";
//        }
//        std::cerr << "}\n";
//    }
//    std::cerr << std::endl;

    {
        const lj::force_calculator  calc_f(ps.device_positions.data().get(),
                                           grid.verlet_list.data().get(),
                                           grid.number_of_neighbors.data().get(),
                                           grid.stride,
                                           boundary);
        thrust::transform(thrust::counting_iterator<std::size_t>(0),
                          thrust::counting_iterator<std::size_t>(N),
                          ps.device_forces.begin(), calc_f);
    }

    const lj::velocity_verlet_update_1 update1(dt, boundary);
    const lj::velocity_verlet_update_2 update2(dt);

    std::cout << "time\tkinetic\tpotential\ttotal\n";
    for(std::size_t s=0; s < step; ++s)
    {
        /* if(s % 100 == 0) */
        {
            const lj::energy_calculator calc_e(
                    ps.device_positions.data().get(),
                    grid.verlet_list.data().get(),
                    grid.number_of_neighbors.data().get(),
                    grid.stride,
                    boundary);

            const float Ek = lj::calc_kinetic_energy(ps);
            const float Ep = thrust::transform_reduce(
                    thrust::counting_iterator<std::size_t>(0),
                    thrust::counting_iterator<std::size_t>(N),
                    calc_e, 0.0, thrust::plus<float>());

            std::cout << s * dt << '\t' << Ek << '\t' << Ep << '\t'
                      << Ek + Ep << '\n';

            ps.pull_device_particles();

            std::ofstream traj("traj.xyz", std::ios_base::app | std::ios_base::out);
            traj << N << "\n\n";
            for(std::size_t i=0; i<N; ++i)
            {
                const auto& v = ps.host_positions[i];
                traj << "H      "
                     << std::fixed << std::setprecision(5) << std::showpoint
                     << std::setw(10) << std::right << v.x
                     << std::setw(10) << std::right << v.y
                     << std::setw(10) << std::right << v.z << '\n';
            }

            std::ofstream velo("velo.xyz", std::ios_base::app | std::ios_base::out);
            velo << N << "\n\n";
            for(std::size_t i=0; i<N; ++i)
            {
                const auto& v = ps.host_velocities[i];
                velo << "H      "
                     << std::fixed << std::setprecision(5) << std::showpoint
                     << std::setw(10) << std::right << v.x
                     << std::setw(10) << std::right << v.y
                     << std::setw(10) << std::right << v.z << '\n';
            }
        }

        // get max(v(t)).
        const float maxv = lj::length(*(thrust::max_element(
                ps.device_velocities.cbegin(), ps.device_velocities.cend(),
                lj::velocity_size_comparator())));

        // p become p(t + dt), v become v(t + dt/2), f become zero here
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                ps.device_masses.begin(),     ps.device_positions.begin(),
                ps.device_velocities.begin(), ps.device_forces.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                ps.device_masses.end(),     ps.device_positions.end(),
                ps.device_velocities.end(), ps.device_forces.end())),
            update1);

        // update cell-list
        grid.update(ps.device_positions, 2 * maxv * dt);

        const lj::force_calculator calc_f(ps.device_positions.data().get(),
                                          grid.verlet_list.data().get(),
                                          grid.number_of_neighbors.data().get(),
                                          grid.stride,
                                          boundary);

        // calculate f(t+dt) using p(t+dt).
        thrust::transform(thrust::counting_iterator<std::size_t>(0),
                          thrust::counting_iterator<std::size_t>(N),
                          ps.device_forces.begin(), calc_f);

        cudaDeviceSynchronize();

        // v become v(t + dt) here
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(
                ps.device_masses.begin(),
                ps.device_velocities.begin(), ps.device_forces.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                ps.device_masses.end(),
                ps.device_velocities.end(), ps.device_forces.end())),
            update2);
    }

    {
        const lj::energy_calculator calc_e(ps.device_positions.data().get(),
                                           grid.verlet_list.data().get(),
                                           grid.number_of_neighbors.data().get(),
                                           grid.stride,
                                           boundary);

        const float Ek = lj::calc_kinetic_energy(ps);
        const float Ep = thrust::transform_reduce(
                thrust::counting_iterator<std::size_t>(0),
                thrust::counting_iterator<std::size_t>(N),
                calc_e, 0.0, thrust::plus<float>());

        std::cout << step * dt << '\t' << Ek << '\t' << Ep << '\t'
                  << Ek + Ep << '\n';

        ps.pull_device_particles();

        std::ofstream traj("traj.xyz", std::ios_base::app | std::ios_base::out);
        traj << N << "\n\n";
        for(std::size_t i=0; i<N; ++i)
        {
            const auto& v = ps.host_positions[i];
            traj << "H      "
                 << std::fixed << std::setprecision(5) << std::showpoint
                 << std::setw(10) << std::right << v.x
                 << std::setw(10) << std::right << v.y
                 << std::setw(10) << std::right << v.z << '\n';
        }

        std::ofstream velo("velo.xyz", std::ios_base::app | std::ios_base::out);
        velo << N << "\n\n";
        for(std::size_t i=0; i<N; ++i)
        {
            const auto& v = ps.host_velocities[i];
            velo << "H      "
                 << std::fixed << std::setprecision(5) << std::showpoint
                 << std::setw(10) << std::right << v.x
                 << std::setw(10) << std::right << v.y
                 << std::setw(10) << std::right << v.z << '\n';
        }
    }
    return 0;
}
