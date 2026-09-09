// SPDX-FileCopyrightText: 2025 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  constexpr size_t N = 100;
  Kokkos::View<double*> x("x", N);
  Kokkos::View<double*> y("y", N);

  Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i) {
    x(i) = (i + 1) * 2.4;
    y(i) = (i + 1) * -1.2;
  });

  Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i) {
    y(i) += 0.5 * x(i);
  });

  Kokkos::View<double> y_0("y_0", 1);
  Kokkos::deep_copy(y_0, y(0));
  Kokkos::View<double> y_N1("y_N1", 1);
  Kokkos::deep_copy(y_N1, y(N-1));

  printf("First and last element (both should be zero): %f, %f\n", y_0(0), y_N1(0));

  Kokkos::finalize();
  return 0;
}
