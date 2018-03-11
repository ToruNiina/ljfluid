#ifndef LJ_PARTICLE
#define LJ_PARTICLE
#include <cuda/vec.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <iostream>
#include <iomanip>

namespace lj
{

struct particle_view
{
    float&  mass()     const noexcept {return *mass_;}
    float4& position() const noexcept {return *position_;}
    float4& velocity() const noexcept {return *velocity_;}
    float4& force()    const noexcept {return *force_;}

    float*  mass_;
    float4* position_;
    float4* velocity_;
    float4* force_;
};

struct const_particle_view
{
    float  const& mass()     const noexcept {return *mass_;}
    float4 const& position() const noexcept {return *position_;}
    float4 const& velocity() const noexcept {return *velocity_;}
    float4 const& force()    const noexcept {return *force_;}

    float  const* mass_;
    float4 const* position_;
    float4 const* velocity_;
    float4 const* force_;
};

struct particle_container
{
    particle_container(const std::size_t N)
        : host_masses(N),       host_positions(N),
          host_velocities(N),   host_forces(N, make_float4(0., 0., 0., 0.)),
          device_masses(N),     device_positions(N),
          device_velocities(N), device_forces(N, make_float4(0., 0., 0., 0.)),
          buf_forces(N)
    {}

    struct vec_add
    {
        __host__ __device__
        float4 operator()(const float4& lhs, const float4& rhs) const noexcept
        {
            return lj::operator+(lhs, rhs);
        }
    };

    void push_host_force()
    {
        this->buf_forces = this->host_forces;
        thrust::transform(buf_forces.begin(), buf_forces.end(),
            device_forces.begin(), device_forces.begin(), vec_add());
    }

    void pull_device_particles()
    {
        this->host_positions  = this->device_positions;
        this->host_velocities = this->device_velocities;
        return;
    }

    __host__
    particle_view operator[](std::size_t i) noexcept
    {
        return particle_view{std::addressof(host_masses[i]),
                             std::addressof(host_positions[i]),
                             std::addressof(host_velocities[i]),
                             std::addressof(host_forces[i])};
    }

    __host__
    const_particle_view operator[](std::size_t i) const noexcept
    {
        return const_particle_view{std::addressof(host_masses[i]),
                                   std::addressof(host_positions[i]),
                                   std::addressof(host_velocities[i]),
                                   std::addressof(host_forces[i])};
    }

    thrust::host_vector<float>  host_masses;
    thrust::host_vector<float4> host_positions;
    thrust::host_vector<float4> host_velocities;
    thrust::host_vector<float4> host_forces;

    thrust::device_vector<float>  device_masses;
    thrust::device_vector<float4> device_positions;
    thrust::device_vector<float4> device_velocities;
    thrust::device_vector<float4> device_forces;
    thrust::device_vector<float4> buf_forces;
};

} // lj
#endif
