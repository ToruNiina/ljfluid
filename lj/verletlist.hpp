#ifndef LJ_VERLET_LIST_HPP
#define LJ_VERLET_LIST_HPP
#include <lj/particle.hpp>
#include <lj/boundary_condition.hpp>

namespace lj
{
template<typename Real>
struct verlet_list
{
    struct range
    {
        using iterator = std::vector<std::size_t>::const_iterator;
        const iterator first, last;
        iterator begin() const noexcept {return first;}
        iterator end()   const noexcept {return last;}
    };

    verlet_list(const Real dt, const Real rc, const Real mgn)
        : dt(dt), cutoff(rc), mergin(mgn), current_mergin(-1.0)
    {}

    void make(const std::vector<particle<Real>>& ps,
              const periodic_boundary<Real>& b)
    {
        indices.clear();
        partners.clear();
        partners.resize(ps.size());

        const Real ignore_distance = cutoff * (1.0 + mergin);
        this->current_mergin = cutoff * mergin;

        std::size_t last = 0;
        for(std::size_t i=0; i<ps.size(); ++i)
        {
            const std::size_t first = last;
            for(std::size_t j=i+1; j < ps.size(); ++j)
            {
                const Real dist =
                    length(b.adjust_direction(ps[i].position - ps[j].position));
                if(dist < ignore_distance)
                {
                    indices.push_back(j);
                    ++last;
                }
            }
            partners[i] = {first, last};
        }
        return;
    }
    void update(const std::vector<particle<Real>>& ps,
                const periodic_boundary<Real>& b, const Real max_vel)
    {
        this->current_mergin -= max_vel * dt;
        if(this->current_mergin < 0.0)
        {
            this->make(ps, b);
        }
        return;
    }

    range neighbors(const std::size_t i) const noexcept
    {
        return range{indices.begin() + partners[i].first,
                     indices.begin() + partners[i].second};
    }

    const Real dt;
    const Real cutoff;
    const Real mergin;
    Real current_mergin;
    std::vector<std::size_t> indices;
    std::vector<std::pair<std::size_t, std::size_t>> partners;
};

} // lj
#endif
