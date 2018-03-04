#ifndef LJ_CELL_LIST_HPP
#define LJ_CELL_LIST_HPP
#include <lj/range.hpp>
#include <lj/particle.hpp>
#include <lj/boundary_condition.hpp>
#include <array>
#include <cassert>

namespace lj
{

template<typename Real>
struct cell_list
{
    cell_list(const Real dt, const Real r_c, const Real mgn,
              const periodic_boundary<Real>& b)
        : dt(dt), cutoff(r_c), mergin(mgn), current_mergin(-1.0),
          lower(b.lower), upper(b.upper)
    {
        const Real cell_size = r_c + (1.0 + this->mergin);
        this->Nx = std::max<std::size_t>(3, std::floor(b.width.x / cell_size));
        this->Ny = std::max<std::size_t>(3, std::floor(b.width.y / cell_size));
        this->Nz = std::max<std::size_t>(3, std::floor(b.width.z / cell_size));

        this->rx = Nx / b.width.x; // == 1.0 / (grid width in x coordinate)
        this->ry = Ny / b.width.y;
        this->rz = Nz / b.width.z;

        this->cells.resize(Nx * Ny * Nz);

        for(std::size_t z=0; z<Nx; ++z)
        {
            for(std::size_t y=0; y<Nx; ++y)
            {
                for(std::size_t x=0; x<Nx; ++x)
                {
                    auto& adjacents = cells.at(calc_index(x, y, z)).first;
                    const std::size_t x_prev = (x ==    0) ? Nx-1 : x-1;
                    const std::size_t x_next = (x == Nx-1) ?    0 : x+1;
                    const std::size_t y_prev = (y ==    0) ? Ny-1 : y-1;
                    const std::size_t y_next = (y == Ny-1) ?    0 : y+1;
                    const std::size_t z_prev = (z ==    0) ? Nz-1 : z-1;
                    const std::size_t z_next = (z == Nz-1) ?    0 : z+1;

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
                }
            }
        }
    }

    void make(const std::vector<particle<Real>>& ps,
              const periodic_boundary<Real>& b)
    {
        indices.clear();

        partners.clear();
        partners.resize(ps.size());

        pidx_cidx_pair.resize(ps.size());

        for(std::size_t i=0; i<ps.size(); ++i)
        {
            pidx_cidx_pair[i] = std::make_pair(i, calc_index(ps[i].position));
        }
        std::sort(pidx_cidx_pair.begin(), pidx_cidx_pair.end(),
                [](const auto& lhs, const auto& rhs) noexcept -> bool {
                    return lhs.second < rhs.second;
                });
        {
            auto iter = pidx_cidx_pair.cbegin();
            for(std::size_t i=0; i<cells.size(); ++i)
            {
                if(iter == pidx_cidx_pair.end() || i != iter->second)
                {
                    cells[i].second = make_range(iter, iter);
                    continue;
                }
                const auto first = iter;
                while(iter != pidx_cidx_pair.cend() && i == iter->second)
                {
                    ++iter;
                }
                cells[i].second = make_range(first, iter);
            }
        }

        const Real threshold2 = std::pow(cutoff * (1.0 + mergin), 2);
        this->current_mergin = cutoff * mergin;

        std::size_t last = 0;
        for(std::size_t i=0; i<ps.size(); ++i)
        {
            const auto& pos1 = ps[i].position;
            const auto& cell = cells[calc_index(pos1)];

            const std::size_t first = last;
            for(const auto cidx : cell.first)
            {
                for(const auto pici : cells[cidx].second)
                {
                    assert(pici.second == cidx);
                    const auto  j = pici.first;
                    if(j <= i) {continue;}
                    const auto& pos2 = ps[j].position;
                    const Real dist2 = length_sq(b.adjust_direction(pos1 - pos2));
                    if(dist2 < threshold2)
                    {
                        indices.push_back(j);
                        ++last;
                    }
                }
            }
            partners[i] = {first, last};
            std::sort(indices.begin() + first, indices.begin() + last);
        }
        return;
    }
    void update(const std::vector<particle<Real>>& ps,
                const periodic_boundary<Real>& b, const Real max_vel)
    {
        if(this->current_mergin < 0.0)
        {
            this->make(ps, b); // cell list at time t (now)
        }
        this->current_mergin -= 2 * max_vel * dt; // mergin at t + dt
        return;
    }

    range<typename std::vector<std::size_t>::const_iterator>
    neighbors(const std::size_t i) const noexcept
    {
        return make_range(indices.begin() + partners[i].first,
                          indices.begin() + partners[i].second);
    }

    std::size_t calc_index(const std::size_t i, const std::size_t j,
                           const std::size_t k) const noexcept
    {
        return i + Nx * j + Nx * Ny * k;
    }

    std::size_t calc_index(const vector<Real>& pos) const noexcept
    {
        const std::size_t ix = std::floor((pos.x - lower.x) * rx);
        const std::size_t iy = std::floor((pos.y - lower.y) * ry);
        const std::size_t iz = std::floor((pos.z - lower.z) * rz);
        return this->calc_index(ix, iy, iz);
    }

    const Real dt;
    const Real cutoff;
    const Real mergin;
    Real current_mergin;
    const vector<Real> lower, upper;
    Real        rx, ry, rz;
    std::size_t Nx, Ny, Nz;

    std::vector<std::pair<std::size_t, std::size_t>> pidx_cidx_pair;
    std::vector<std::pair<
        std::array<std::size_t, 27>,
        range<typename decltype(pidx_cidx_pair)::const_iterator>
        >> cells;
    std::vector<std::size_t> indices;
    std::vector<std::pair<std::size_t, std::size_t>> partners;
};

} // lj
#endif
