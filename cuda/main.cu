#include <cuda/particle.cuh>
#include <cuda/boundary_condition.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <curand.h>
#include <iterator>
#include <cassert>
#include <cmath>

namespace lj
{
__device__ __host__ constexpr float sgm() noexcept {return 1.0f;}
__device__ __host__ constexpr float eps() noexcept {return 1.0f;}

float kinetic_energy(const thrust::device_vector<particle>& ps)
{
    return thrust::transform_reduce(ps.cbegin(), ps.cend(),
            [] __device__ (const particle& p) -> float {
                return p.mass * lj::length_sq(p.velocity) * 0.5;
            }, 0.0, thrust::plus<float>());
}

} // lj

struct initializer
{
    initializer(thrust::device_ptr<float> p) : boltz_ptr(p){}

    __device__
    lj::particle operator()(const std::size_t i)
    {
        lj::particle p;
        p.mass = 1.0;
        p.position = make_float4(
            5.0 + 10.0 * ((i & 0b000000001111) >> 0),
            5.0 + 10.0 * ((i & 0b000011110000) >> 4),
            5.0 + 10.0 * ((i & 0b111100000000) >> 8),
            0.0);
        p.velocity = make_float4(
            *(boltz_ptr + 3*i  ),
            *(boltz_ptr + 3*i+1),
            *(boltz_ptr + 3*i+2),
            0.0);
        p.force = make_float4(0.0, 0.0, 0.0, 0.0);
        return p;
    }

    thrust::device_ptr<float> boltz_ptr;
};

int main()
{
    const float4 upper    = make_float4(160.0, 160.0, 160.0, 0.0);
    const float4 lower    = make_float4(  0.0,   0.0,   0.0, 0.0);
    const auto   boundary = lj::make_boundary(lower, upper);

    const std::size_t N    = std::pow(16, 3);
    const std::size_t seed = 123456789;
    const float kB  = 1.986231313e-3;
    const float T   = 300.0;
    const float dt  = 0.01;

    thrust::device_vector<lj::particle> ps(N);

    /* initialization */{
        // prepair cuRAND generators
        curandGenerator_t rng;
        const auto st_gen = curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
        assert(st_gen  == CURAND_STATUS_SUCCESS);
        const auto st_seed = curandSetPseudoRandomGeneratorSeed(rng, seed);
        assert(st_seed == CURAND_STATUS_SUCCESS);

        thrust::device_vector<float> boltz(3 * N);
        const auto st_genrnd = curandGenerateNormal(rng, boltz.data().get(), 3 * N,
                /*mean = */ 0.0, /*stddev = */ std::sqrt(kB * T));
        assert(st_genrnd == CURAND_STATUS_SUCCESS);

        thrust::transform(thrust::make_counting_iterator<std::size_t>(0),
                          thrust::make_counting_iterator<std::size_t>(N),
                          ps.begin(), initializer(boltz.data()));
    }

    std::cout << ps << std::endl;
    std::cerr << "kinetic energy = " << lj::kinetic_energy(ps)
              << ", 3/2 NkBT = " << N * kB * T * 1.5 << std::endl;

    //TODO



    return 0;
}
