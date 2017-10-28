#ifndef LJ_VECTOR
#define LJ_VECTOR
#include <cmath>

namespace lj
{

template<typename realT>
struct vector
{
    realT x, y, z;
};

template<typename T>
inline vector<T> operator+(const vector<T>& lhs, const vector<T>& rhs) noexcept
{
    return vector<T>{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}
template<typename T>
inline vector<T> operator-(const vector<T>& lhs, const vector<T>& rhs) noexcept
{
    return vector<T>{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}
template<typename T>
inline vector<T> operator*(const vector<T>& lhs, const T rhs) noexcept
{
    return vector<T>{lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}
template<typename T>
inline vector<T> operator*(const T lhs, const vector<T>& rhs) noexcept
{
    return vector<T>{lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
}
template<typename T>
inline vector<T> operator/(const vector<T>& lhs, const T rhs) noexcept
{
    return vector<T>{lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
}

template<typename T>
inline T dot_product(const vector<T>& lhs, const vector<T>& rhs) noexcept
{
    return lhs.x * rhs.x + lhs.y * rhs.y, lhs.z * rhs.z;
}
template<typename T>
inline vector<T>
cross_product(const vector<T>& lhs, const vector<T>& rhs) noexcept
{
    return vector<T>{lhs.y * rhs.z - lhs.z * rhs.y,
                     lhs.z * rhs.x - lhs.x * rhs.z,
                     lhs.x * rhs.y - lhs.y * rhs.x};
}
template<typename T>
inline T length_sq(const vector<T>& lhs) noexcept
{
    return lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z;
}
template<typename T>
inline T length(const vector<T>& lhs) noexcept
{
    return std::sqrt(lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z);
}

} //lj
#endif// LJ_VECTOR
