#ifndef LJ_ARRAY_H
#define LJ_ARRAY_H

namespace lj
{

template<typename T, std::size_t N>
struct array
{
    typedef T           value_type;
    typedef std::size_t size_type;
    typedef value_type&       reference;
    typedef value_type const& const_reference;
    typedef value_type*       pointer;
    typedef value_type const* const_pointer;

    __host__ __device__
    reference       operator[](std::size_t i)       noexcept {return data_[i];}
    __host__ __device__
    const_reference operator[](std::size_t i) const noexcept {return data_[i];}

    __host__ __device__ reference       front()       noexcept {return data_[0];}
    __host__ __device__ const_reference front() const noexcept {return data_[0];}
    __host__ __device__ reference       back()        noexcept {return data_[N-1];}
    __host__ __device__ const_reference back()  const noexcept {return data_[N-1];}

    __host__ __device__ bool        empty()    const noexcept {return N==0;}
    __host__ __device__ std::size_t size()     const noexcept {return N;}
    __host__ __device__ std::size_t max_size() const noexcept {return N;}

    __host__ __device__ void fill(T v) noexcept
    {
        for(std::size_t i=0; i<N; ++i)
        {
            data_[i] = v;
        }
        return;
    }

    __host__ __device__ void swap(array<T, N>& rhs) noexcept
    {
        using std::swap; using thrust::swap;
        for(std::size_t i=0; i<N; ++i)
        {
            swap(data_[i], rhs.data_[i]);
        }
        return;
    }

    __host__ reference at(std::size_t i)
    {if(i >= N) throw std::out_of_range("lj::array::at"); else return data_[i];}
    __host__ const_reference at(std::size_t i) const
    {if(i >= N) throw std::out_of_range("lj::array::at"); else return data_[i];}

    T data_[N];
};



} // lj
#endif// LJ_ARRAY_H
