// Example of compile time calculation over an array
//
// Compile with:
// 
//     g++ -std=c++26 -o constexpr constexpr.cpp -O3
//

#include <cmath>
#include <array>
#include <numeric>
#include <algorithm>

constexpr double myfunc(double q, double a, double b, double c) {
  // sqrt() is constexpr since C++26
  return a * q * q + b * q + c * std::sqrt(b);
}

template <size_t N>
constexpr std::array<double, N> calc_intensities(const std::array<double, N>& q) {
    std::array<double, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = myfunc(q[i], 2.3, 4.2, -0.5);
    }
    return result;

}
int main() {
  // it is customary to CAPITALIZE constexpr variables
  constexpr auto Q = std::array{10.0, 20.0, 3.0, 4.0, 5.0, -2.0};
  constexpr auto I = calc_intensities(Q);
  constexpr auto SUM = std::ranges::fold_left(I, 0.0, std::plus<>());
  return int(SUM);
}
