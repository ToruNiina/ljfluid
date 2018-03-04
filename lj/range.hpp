#ifndef LJ_RANGE_HPP
#define LJ_RANGE_HPP

namespace lj
{

template<typename Iterator>
struct range
{
    Iterator first, last;
    Iterator begin() const noexcept {return first;}
    Iterator end()   const noexcept {return last;}
};

template<typename Iterator>
range<Iterator> make_range(Iterator first, Iterator last)
{
    return range<Iterator>{first, last};
}

}
#endif //LJ_RANGE_HPP
