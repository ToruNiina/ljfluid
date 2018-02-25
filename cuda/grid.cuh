#ifndef LJ_GRID_H
#define LJ_GRID_H
#include <cuda/array.h>
#include <cuda/particle.cuh>
#include <cuda/boundary_condition.cuh>
#include <thrust/device_vector.h>
#include <thrust/zip_iterator.h>

namespace lj
{

struct grid
{
    __host__
    grid(const float rc, periodic_boundary b): boundary(b)
    {
        this->Nx = std::floor(this->boundary.width.x / rc);
        this->Ny = std::floor(this->boundary.width.y / rc);
        this->Nz = std::floor(this->boundary.width.z / rc);

        this->rx = Nx / boundary.width.x; // == 1.0 / grid_x_width
        this->ry = Ny / boundary.width.y;
        this->rz = Nz / boundary.width.z;

        this->adjs.resize(Nx * Ny * Nz);
        this->cell.resize(Nx * Ny * Nz);

        for(std::size_t k=0; k<Nz; ++k)
        {
            for(std::size_t j=0; j<Ny; ++j)
            {
                for(std::size_t i=0; i<Nx; ++i)
                {
                    adjs[calc_index(i, j, k)] = make_adjacents(i, j, k);
                }
            }
        }
    }

    __host__
    void assign(const thrust::device_vector<particle>& ps)
    {
        cellids.resize(ps.size());
        idxs.resize(ps.size());

        thrust::copy(thrust::make_counting_iterator<std::size_t>(0),
                     thrust::make_counting_iterator<std::size_t>(idxs.size()),
                     idxs.begin()); // idxs = {0, 1, 2, 3, ..., ps.size()}

        thrust::transform(ps.begin(), ps.end(), cellids.begin(),
            [this, rx, ry, rz] __device__ (const particle& p) -> std::size_t {
                return this->calc_index(p.position);
            });

        thrust::stable_sort_by_key(cellids.begin(), cellids.end(), idxs.begin());
        // cellids = {0, 0, 1, 1, 1, 3, 3, 4, ... } // cell ids
        // idxs    = {6, 9, 2, 5, 7, 1, 4, 3, ... } // particle idx in the cell
        // cell   -> {2, 3, 0, 2, ...} // number of particles in each cell

        //XXX avoid empty cell problem (in the above case, cell#2 has nothing)
        thrust::device_vector<std::size_t> filled_grids(cell.size());
        thrust::device_vector<std::size_t> num_in_cell (cell.size());
        thrust::fill(cell.begin(), cell.end(), 0);

        // {filled_grid.end, num_in_cell.end}
        const auto ends = thrust::reduce_by_key(
            /* key range  = */ cellids.begin(), cellids.end(),
            /* weight     = */ thrust::make_constant_iterator(1),
            /* key result = */ filled_grids.begin(),
            /* ps / grid  = */ num_in_cell.begin());

        thrust::gather(filled_grids.begin(), ends.first,
                       num_in_cell.begin(), cell.begin());
        return;
    }

    __host__ __device__
    std::size_t calc_index(const float4& pos) const
    {
        const std::size_t ix = floorf(pos.x * rx);
        const std::size_t iy = floorf(pos.y * ry);
        const std::size_t iz = floorf(pos.z * rz);
        return this->calc_index(ix, iy, iz);
    }

    float       rc;
    float       rx, ry, rz;
    std::size_t Nx, Ny, Nz;
    periodic_boundary boundary;
    thrust::device_vector<std::size_t> cellids; // tmp
    thrust::device_vector<array<std::size_t, 27>> adjs;
    thrust::device_vector<std::size_t> idxs; // list of indices sorted by cell id
    thrust::device_vector<std::size_t> cell; // number of particles in each cell

  private:

    __host__
    array<std::size_t, 27> make_adjacents(
            const std::size_t x, const std::size_t y, const std::size_t z) const
    {
        array<std::size_t, 27> adjacents;
        adjacents[ 0] = calc_index((x  == 0 ? Nx : x-1), (y == 0 ? Ny : y-1), (z==0 ? Nz : z-1));
        adjacents[ 1] = calc_index( x,                   (y == 0 ? Ny : y-1), (z==0 ? Nz : z-1));
        adjacents[ 2] = calc_index((x+1==Nx ?  0 : x+1), (y == 0 ? Ny : y-1), (z==0 ? Nz : z-1));
        adjacents[ 3] = calc_index((x  == 0 ? Nx : x-1), y,                   (z==0 ? Nz : z-1));
        adjacents[ 4] = calc_index( x,                   y,                   (z==0 ? Nz : z-1));
        adjacents[ 5] = calc_index((x+1==Nx ?  0 : x+1), y,                   (z==0 ? Nz : z-1));
        adjacents[ 6] = calc_index((x  == 0 ? Nx : x-1), (y+1==Ny ? 0 : y+1), (z==0 ? Nz : z-1));
        adjacents[ 7] = calc_index( x,                   (y+1==Ny ? 0 : y+1), (z==0 ? Nz : z-1));
        adjacents[ 8] = calc_index((x+1==Nx ?  0 : x+1), (y+1==Ny ? 0 : y+1), (z==0 ? Nz : z-1));

        adjacents[ 9] = calc_index((x  == 0 ? Nx : x-1), (y == 0 ? Ny : y-1), z);
        adjacents[10] = calc_index( x,                   (y == 0 ? Ny : y-1), z);
        adjacents[11] = calc_index((x+1==Nx ?  0 : x+1), (y == 0 ? Ny : y-1), z);
        adjacents[12] = calc_index((x  == 0 ? Nx : x-1), y,                   z);
        adjacents[13] = calc_index( x,                   y,                   z);
        adjacents[14] = calc_index((x+1==Nx ?  0 : x+1), y,                   z);
        adjacents[15] = calc_index((x  == 0 ? Nx : x-1), (y+1==Ny ? 0 : y+1), z);
        adjacents[16] = calc_index( x,                   (y+1==Ny ? 0 : y+1), z);
        adjacents[17] = calc_index((x+1==Nx ?  0 : x+1), (y+1==Ny ? 0 : y+1), z);

        adjacents[18] = calc_index((x  == 0 ? Nx : x-1), (y == 0 ? Ny : y-1), (z+1==Nz ? 0 : z+1));
        adjacents[19] = calc_index( x,                   (y == 0 ? Ny : y-1), (z+1==Nz ? 0 : z+1));
        adjacents[20] = calc_index((x+1==Nx ?  0 : x+1), (y == 0 ? Ny : y-1), (z+1==Nz ? 0 : z+1));
        adjacents[21] = calc_index((x  == 0 ? Nx : x-1), y,                   (z+1==Nz ? 0 : z+1));
        adjacents[22] = calc_index( x,                   y,                   (z+1==Nz ? 0 : z+1));
        adjacents[23] = calc_index((x+1==Nx ?  0 : x+1), y,                   (z+1==Nz ? 0 : z+1));
        adjacents[24] = calc_index((x  == 0 ? Nx : x-1), (y+1==Ny ? 0 : y+1), (z+1==Nz ? 0 : z+1));
        adjacents[25] = calc_index( x,                   (y+1==Ny ? 0 : y+1), (z+1==Nz ? 0 : z+1));
        adjacents[26] = calc_index((x+1==Nx ?  0 : x+1), (y+1==Ny ? 0 : y+1), (z+1==Nz ? 0 : z+1));

        return adjacents;
    }

    __host__ __device__
    std::size_t calc_index(
            const std::size_t i, const std::size_t j, const std::size_t k) const
    {
        return k * Ny * Nx + j * Nx + i;
    }
};

} // lj
#endif// LJ_GRID_H
