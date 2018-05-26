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
#include <thrust/extrema.h>
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

__device__ __host__
void sort(std::size_t* const region, const std::size_t len)
{
    if(len < 2){return;}
    for(std::size_t i=0; i<len-1; ++i)
    {
        for(std::size_t j=i+1; j<len; ++j)
        {
            if(region[i] > region[j])
            {
                std::size_t tmp = region[i];
                region[i] = region[j];
                region[j] = tmp;
            }
        }
    }
}

} // detail

struct grid
{
    struct index_calculator : thrust::unary_function<float4, std::size_t>
    {
        __host__
        index_calculator(const float rx_, const float ry_, const float rz_,
             const std::size_t Nx_, const std::size_t Ny_, const std::size_t Nz_,
             const float4 low) noexcept
            : rx(rx_), ry(ry_), rz(rz_), Nx(Nx_), Ny(Ny_), Nz(Nz_),
              low_x(low.x), low_y(low.y), low_z(low.z)
        {}

        __host__ __device__
        std::size_t operator()(float4 pos) const noexcept
        {
            std::size_t i = floorf((pos.x - low_x) * rx);
            std::size_t j = floorf((pos.y - low_y) * ry);
            std::size_t k = floorf((pos.z - low_z) * rz);
            if(i == Nx) {i = 0;}
            if(j == Ny) {j = 0;}
            if(k == Nz) {k = 0;}
            return k * Ny * Nx + j * Nx + i;
        }

        const float       rx, ry, rz;
        const float       low_x, low_y, low_z;
        const std::size_t Nx, Ny, Nz;
    };

    struct verlet_list_generator
    {
        __host__
        verlet_list_generator(
                const float threshold_, const std::size_t stride_,
                std::size_t* const vlist, std::size_t* const n_neigh,
                const float4* const ps, const array<std::size_t, 27>* const adjs_,
                const std::size_t* const idxs_, const std::size_t* const clist_,
                index_calculator calc_idx, periodic_boundary bdry)
            : threshold2(threshold_ * threshold_), stride(stride_),
              verlet_list(vlist), num_neighbor(n_neigh), positions(ps),
              adjs(adjs_), indices(idxs_), cell_list(clist_),
              calc_index(calc_idx),  boundary(bdry)
        {}

        __host__ __device__
        void operator()(const std::size_t i) const
        {
            std::size_t* ptr_vlist = verlet_list + i * stride;
            const float4      pos1 = *(positions + i);
            const std::size_t cidx = calc_index(pos1);
            const array<std::size_t, 27>& adjacents = *(adjs + cidx);

            std::size_t n_neigh = 0;
            for(std::size_t i_neigh=0; i_neigh<27; ++i_neigh)
            {
                // index of adjacent cell
                const std::size_t cell_idx = adjacents[i_neigh];
                const std::size_t first = *(cell_list + cell_idx);
                const std::size_t last  = *(cell_list + cell_idx + 1);

                for(std::size_t pi = first; pi < last; ++pi)
                {
                    // index of possible partner
                    const std::size_t pidx = indices[pi];
                    if(pidx == i) {continue;}
                    const float4 pos2 = positions[pidx];
                    const float dist2 = length_sq(
                            adjust_direction(pos1 - pos2, boundary));
                    if(dist2 < threshold2)
                    {
                        *ptr_vlist = pidx;
                        ++ptr_vlist;
                        ++n_neigh;
                    }
                }
            }
            *(num_neighbor + i) = n_neigh;
            detail::sort(verlet_list + i*stride, n_neigh);
            assert(n_neigh < stride);
            return ;
        }

        const float      threshold2;
        const std::size_t   stride;
        std::size_t*  const verlet_list;
        std::size_t*  const num_neighbor;
        const float4* const positions;
        const array<std::size_t, 27>* const adjs;
        const std::size_t* const indices;
        const std::size_t* const cell_list;
        const index_calculator   calc_index;
        const periodic_boundary boundary;
    };

    __host__
    grid(const float rc_, const float mergin_, const periodic_boundary b)
        : rc(rc_), mergin(mergin_), current_mergin(-1), boundary(b)
    {
        this->Nx = std::max<std::size_t>(3, std::floor(b.width.x / rc));
        this->Ny = std::max<std::size_t>(3, std::floor(b.width.y / rc));
        this->Nz = std::max<std::size_t>(3, std::floor(b.width.z / rc));

        this->rx = Nx / b.width.x; // == 1.0 / grid_x_width (reciprocal grid x)
        this->ry = Ny / b.width.y;
        this->rz = Nz / b.width.z;

        this->adjs.resize(Nx * Ny * Nz);
        this->cell.resize(Nx * Ny * Nz + 1);

        this->cellid_of_bins.resize(Nx * Ny * Nz);
        this->tmp_number_of_particles.resize(Nx * Ny * Nz);
        this->number_of_particles_in_cell.resize(Nx * Ny * Nz);

        thrust::transform(
            thrust::make_counting_iterator<std::size_t>(0),
            thrust::make_counting_iterator<std::size_t>(Nx * Ny * Nz),
            this->adjs.begin(), detail::make_adjacents(Nx, Ny, Nz));
    }


    void update(const thrust::device_vector<float4>& ps,
                const float max_displacement)
    {
        current_mergin -= max_displacement;
        if(current_mergin < 0.0)
        {
            // make grid by using p(t+dt)
            this->assign(ps);
            current_mergin = rc * mergin;
        }
        return;
    }

    __host__
    void assign(const thrust::device_vector<float4>& ps)
    {
        number_of_neighbors.resize(ps.size());
        cellid_of_particles.resize(ps.size());

        idxs.resize(ps.size());
        thrust::copy(thrust::make_counting_iterator<std::size_t>(0),
                     thrust::make_counting_iterator<std::size_t>(idxs.size()),
                     idxs.begin());

        // calculate cell id for each particle
        thrust::transform(ps.begin(), ps.end(), cellid_of_particles.begin(),
            index_calculator(rx, ry, rz, Nx, Ny, Nz, boundary.lower));

        // sort particle indices by its cell id
        // ptcl id = {0, 1, 2, 3, ...  N} // particle idx
        // cell id = {1, 3, 5, 2, ... 10} // belonging cell idx
        //  v
        // ptcl id = {0, 4, 6, 3, 5, 1, 7, 8, 9, 10, 12, ...}
        // cell id = {1, 1, 1, 2, 2, 3, 3, 3, 3,  5,  5, ...}
        thrust::stable_sort_by_key(cellid_of_particles.begin(),
                                   cellid_of_particles.end(), idxs.begin());

        // calculate number of particles in each cell
        // ptcl id = {0, 4, 6, 3, 5, 1, 7, 8, 9, 10, 12, ...}
        // cell id = {1, 1, 1, 2, 2, 3, 3, 3, 3,  5,  5, ...}
        //  v
        // cid of bin    = {1, 2, 3, 5, ...}
        // num p in bin  = {3, 2, 4, 2, ...}
       const auto tmp_number_of_particles_end = thrust::reduce_by_key(
                cellid_of_particles.begin(), cellid_of_particles.end(),
                thrust::constant_iterator<std::size_t>(1),
                cellid_of_bins.begin(),
                tmp_number_of_particles.begin(),
                thrust::equal_to<std::size_t>()).second;

        /* {
            thrust::host_vector<std::size_t> host_cellid_of_bins(cellid_of_bins);
            thrust::host_vector<std::size_t> host_num_particles(tmp_number_of_particles);

            std::cerr << "cellids of bins = ";
            for(std::size_t i=0; i<host_cellid_of_bins.size(); ++i)
            {
                std::cerr << host_cellid_of_bins[i] << ", ";
            }
            std::cerr << '\n';
            std::cerr << "num of particles = ";
            for(std::size_t i=0; i<host_num_particles.size(); ++i)
            {
                std::cerr << host_num_particles[i] << ", ";
            }
            std::cerr << '\n';
        } */

        // avoid empty cell problem
        // cid of bin    = {1, 2, 3, 5, ...} <- map // grid #4 is empty!
        // num p in bin  = {3, 2, 4, 2, ...} <- value will be scattered
        //  v
        // cid of bin    = {1, 2, 3, 4, 5, ...}
        // num p in bin  = {3, 2, 4, 0, 2, ...}
        thrust::fill(number_of_particles_in_cell.begin(),
                     number_of_particles_in_cell.end(), 0);
        thrust::scatter(tmp_number_of_particles.begin(),
                        tmp_number_of_particles_end,
                        cellid_of_bins.begin(),
                        number_of_particles_in_cell.begin());

        /* {
            thrust::host_vector<std::size_t> host_cellid_of_bins(cellid_of_bins);
            thrust::host_vector<std::size_t> host_num_particles(number_of_particles_in_cell);

            std::cerr << "cellids of bins = ";
            for(std::size_t i=0; i<host_cellid_of_bins.size(); ++i)
            {
                std::cerr << host_cellid_of_bins[i] << ", ";
            }
            std::cerr << '\n';
            std::cerr << "num of particles = ";
            for(std::size_t i=0; i<host_num_particles.size(); ++i)
            {
                std::cerr << host_num_particles[i] << ", ";
            }
            std::cerr << '\n';
        } */

        thrust::inclusive_scan(number_of_particles_in_cell.begin(),
                               number_of_particles_in_cell.end(),
                               cell.begin() + 1);

        // align verlet list
        const std::size_t maxN = *thrust::max_element(
                number_of_particles_in_cell.begin(),
                number_of_particles_in_cell.end());
        this->stride = std::ceil((maxN * 27) / 16) * 16;

        verlet_list.resize(ps.size() * stride);
        thrust::fill(verlet_list.begin(), verlet_list.end(),
                     std::numeric_limits<std::size_t>::max());

        // generate verlet list using grid
        thrust::for_each(
            thrust::counting_iterator<std::size_t>(0),
            thrust::counting_iterator<std::size_t>(ps.size()),
            verlet_list_generator(rc * (1.0f + mergin), stride,
                verlet_list.data().get(), number_of_neighbors.data().get(),
                ps.data().get(), adjs.data().get(), idxs.data().get(),
                cell.data().get(),
                index_calculator(rx, ry, rz, Nx, Ny, Nz, boundary.lower),
                boundary)
            );
        return;
    }

    const float rc, mergin;
    float       current_mergin;
    float       rx, ry, rz;
    std::size_t Nx, Ny, Nz;
    std::size_t stride;
    periodic_boundary boundary;
    thrust::device_vector<array<std::size_t, 27>> adjs;

    thrust::device_vector<std::size_t> cellid_of_particles;
    thrust::device_vector<std::size_t> cellid_of_bins;
    thrust::device_vector<std::size_t> tmp_number_of_particles;
    thrust::device_vector<std::size_t> number_of_particles_in_cell;

    thrust::device_vector<std::size_t> idxs;
    thrust::device_vector<std::size_t> cell;

    thrust::device_vector<std::size_t> number_of_neighbors;
    thrust::device_vector<std::size_t> verlet_list;
};

} // lj
#endif// LJ_GRID_H
