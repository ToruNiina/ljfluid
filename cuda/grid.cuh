#ifndef LJ_GRID_H
#define LJ_GRID_H
#include <cuda/array.cuh>
#include <cuda/particle.cuh>
#include <cuda/boundary_condition.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>

namespace lj
{
namespace detail
{
struct make_adjacents
    : thrust::unary_function<std::size_t, ::lj::array<std::size_t, 27>>
{
    const std::size_t Nx, Ny, Nz;

    __host__
    make_adjacents(const std::size_t x, const std::size_t y, const std::size_t z)
        : Nx(x), Ny(y), Nz(z)
    {}

    __device__
    ::lj::array<std::size_t, 27> operator()(const std::size_t idx) const noexcept
    {
        const std::size_t x = idx % Nx;
        const std::size_t y = (idx / Nx) % Ny;
        const std::size_t z = (idx / Nx / Ny);

        const std::size_t x_prev = (x ==    0) ? Nx-1 : x-1;
        const std::size_t x_next = (x == Nx-1) ?    0 : x+1;
        const std::size_t y_prev = (y ==    0) ? Ny-1 : y-1;
        const std::size_t y_next = (y == Ny-1) ?    0 : y+1;
        const std::size_t z_prev = (z ==    0) ? Nz-1 : z-1;
        const std::size_t z_next = (z == Nz-1) ?    0 : z+1;

        ::lj::array<std::size_t, 27> adjacents;
        adjacents[ 0] = calc_index(x_prev, y_prev, z_prev);
        adjacents[ 1] = calc_index(x,      y_prev, z_prev);
        adjacents[ 2] = calc_index(x_next, y_prev, z_prev);
        adjacents[ 3] = calc_index(x_prev, y,      z_prev);
        adjacents[ 4] = calc_index(x,      y,      z_prev);
        adjacents[ 5] = calc_index(x_next, y,      z_prev);
        adjacents[ 6] = calc_index(x_prev, y_next, z_prev);
        adjacents[ 7] = calc_index(x,      y_next, z_prev);
        adjacents[ 8] = calc_index(x_next, y_next, z_prev);

        adjacents[ 9] = calc_index(x_prev, y_prev, z);
        adjacents[10] = calc_index(x,      y_prev, z);
        adjacents[11] = calc_index(x_next, y_prev, z);
        adjacents[12] = calc_index(x_prev, y,      z);
        adjacents[13] = calc_index(x,      y,      z);
        adjacents[14] = calc_index(x_next, y,      z);
        adjacents[15] = calc_index(x_prev, y_next, z);
        adjacents[16] = calc_index(x,      y_next, z);
        adjacents[17] = calc_index(x_next, y_next, z);

        adjacents[18] = calc_index(x_prev, y_prev, z_next);
        adjacents[19] = calc_index(x,      y_prev, z_next);
        adjacents[20] = calc_index(x_next, y_prev, z_next);
        adjacents[21] = calc_index(x_prev, y,      z_next);
        adjacents[22] = calc_index(x,      y,      z_next);
        adjacents[23] = calc_index(x_next, y,      z_next);
        adjacents[24] = calc_index(x_prev, y_next, z_next);
        adjacents[25] = calc_index(x,      y_next, z_next);
        adjacents[26] = calc_index(x_next, y_next, z_next);

        return adjacents;
    }

    __device__
    std::size_t calc_index(
        const std::size_t i, const std::size_t j, const std::size_t k) const
    {
        return k * Ny * Nx + j * Nx + i;
    }
};
} // detail

struct grid
{
    struct index_calculator : thrust::unary_function<float4, std::size_t>
    {
        __host__
        index_calculator(const float rx_, const float ry_, const float rz_,
             const std::size_t Nx_, const std::size_t Ny_, const std::size_t Nz_)
            : rx(rx_), ry(ry_), rz(rz_), Nx(Nx_), Ny(Ny_), Nz(Nz_)
        {}

        __host__ __device__
        std::size_t operator()(float4 pos) const noexcept
        {
            const std::size_t i = floorf(pos.x * rx);
            const std::size_t j = floorf(pos.y * ry);
            const std::size_t k = floorf(pos.z * rz);
            return k * Ny * Nx + j * Nx + i;
        }
        const float       rx, ry, rz;
        const std::size_t Nx, Ny, Nz;
    };

    __host__
    grid(const float rc, periodic_boundary b): boundary(b)
    {
        this->Nx = std::max<std::size_t>(3, std::floor(this->boundary.width.x / rc));
        this->Ny = std::max<std::size_t>(3, std::floor(this->boundary.width.y / rc));
        this->Nz = std::max<std::size_t>(3, std::floor(this->boundary.width.z / rc));
        std::cerr << this->Nx << ", " << this->Ny << ", " << this->Nz << std::endl;

        this->rx = Nx / boundary.width.x; // == 1.0 / grid_x_width
        this->ry = Ny / boundary.width.y;
        this->rz = Nz / boundary.width.z;
        std::cerr << this->rx << ", " << this->ry << ", " << this->rz << std::endl;

        this->adjs.resize(Nx * Ny * Nz);
        this->cell.resize(Nx * Ny * Nz + 1);

        this->cellids_of_bins.resize(Nx * Ny * Nz);
        this->tmp_number_of_particles.resize(Nx * Ny * Nz);
        this->number_of_particles_in_cell.resize(Nx * Ny * Nz);

        thrust::transform(
            thrust::make_counting_iterator<std::size_t>(0),
            thrust::make_counting_iterator<std::size_t>(Nx * Ny * Nz),
            this->adjs.begin(), detail::make_adjacents(Nx, Ny, Nz));
    }

    __host__
    void assign(const thrust::device_vector<float4>& ps)
    {
        cellids_of_particles.resize(ps.size());
        idxs.resize(ps.size());

        // initialize indices
        thrust::copy(thrust::make_counting_iterator<std::size_t>(0),
                     thrust::make_counting_iterator<std::size_t>(idxs.size()),
                     idxs.begin()); // idxs = {0, 1, 2, 3, ..., ps.size()-1}

        {
            const thrust::host_vector<std::size_t> hv = idxs;
            for(const auto h : hv)
                std::cerr << h << ' ';
            std::cerr << std::endl;
        }
        std::cerr << "index initialized" << std::endl;

        // calculate cell id for each particle
        thrust::transform(ps.begin(), ps.end(), cellids_of_particles.begin(),
            index_calculator(rx, ry, rz, Nx, Ny, Nz));
        {
            const thrust::host_vector<float4> p = ps;
            const thrust::host_vector<std::size_t> hv = cellids_of_particles;
            for(std::size_t i=0; i<p.size();++i)
            {
                std::cerr << "{" << p[i].x << ", " << p[i].y << ", " << p[i].z << "}, " << hv[i] << ' ';
            }
            std::cerr << std::endl;
        }

        std::cerr << "cell index calculated" << std::endl;

        // sort particle indices by its cell id
        // ptcl id = {0, 1, 2, 3, ...  N} // particle idx
        // cell id = {1, 3, 5, 2, ... 10} // belonging cell idx
        //  |
        //  v
        // ptcl id = {0, 4, 6, 3, 5, 1, 7, 8, 9, 10, 12, ...}
        // cell id = {1, 1, 1, 2, 2, 3, 3, 3, 3,  5,  5, ...}
        thrust::stable_sort_by_key(cellids_of_particles.begin(),
                                   cellids_of_particles.end(), idxs.begin());

        std::cerr << "particle indices are sorted" << std::endl;
        {
            const thrust::host_vector<std::size_t> hv = cellids_of_particles;
            for(const auto h : hv)
                std::cerr << h << ' ';
            std::cerr << std::endl;
        }
        {
            const thrust::host_vector<std::size_t> hv = idxs;
            for(const auto h : hv)
                std::cerr << h << ' ';
            std::cerr << std::endl;
        }

        // calculate number of particles in each cell
        // ptcl id = {0, 4, 6, 3, 5, 1, 7, 8, 9, 10, 12, ...}
        // cell id = {1, 1, 1, 2, 2, 3, 3, 3, 3,  5,  5, ...}
        //  |
        //  v
        // cid of bin    = {1, 2, 3, 5, ...}
        // num p in bin  = {3, 2, 4, 2, ...}
       const auto cidend_npend = thrust::reduce_by_key(
                cellids_of_particles.begin(), cellids_of_particles.end(),
                thrust::constant_iterator<std::size_t>(1),
                cellids_of_bins.begin(),
                tmp_number_of_particles.begin(),
                thrust::equal_to<std::size_t>());

        std::cerr << "count number of particles" << std::endl;

        // avoid empty cell problem
        // cid of bin    = {1, 2, 3, 5, ...}
        // num p in bin  = {3, 2, 4, 2, ...}
        //  |
        //  v
        // cid of bin    = {1, 2, 3, 4, 5, ...}
        // num p in bin  = {3, 2, 4, 0, 2, ...}
        thrust::fill(number_of_particles_in_cell.begin(),
                     number_of_particles_in_cell.end(), 0);
        thrust::gather(cellids_of_bins.begin(),
                       cellids_of_bins.end(),
                       tmp_number_of_particles.begin(),
                       number_of_particles_in_cell.begin());

        std::cerr << "gather number of particles for cells" << std::endl;

        thrust::inclusive_scan(number_of_particles_in_cell.begin(),
                               number_of_particles_in_cell.end(),
                               cell.begin() + 1);

         std::cerr << "indices are scanned" << std::endl;
        return;
    }

    __host__
    std::pair<std::size_t, std::size_t> get_range(std::size_t i) const noexcept
    {
        return std::make_pair(cell[i], cell[i+1]);
    }

    float       rc;
    float       rx, ry, rz;
    std::size_t Nx, Ny, Nz;
    periodic_boundary boundary;
    thrust::device_vector<array<std::size_t, 27>> adjs;

    thrust::device_vector<std::size_t> cellids_of_particles;
    thrust::device_vector<std::size_t> cellids_of_bins;
    thrust::device_vector<std::size_t> tmp_number_of_particles;
    thrust::device_vector<std::size_t> number_of_particles_in_cell;

    thrust::device_vector<std::size_t> idxs; // list of indices sorted by cell id
    thrust::device_vector<std::size_t> cell; // beginning index of idxs for each cell
};

} // lj
#endif// LJ_GRID_H
