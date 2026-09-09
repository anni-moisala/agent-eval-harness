! SPDX-FileCopyrightText: 2025 CSC - IT Center for Science Ltd. <www.csc.fi>
!
! SPDX-License-Identifier: MIT

#include "axpy_helper_functions.F90"

program axpy
  use axpy_helper_functions
  implicit none
  character(len=32) :: arg
  integer :: i, n
  real(8) :: alpha, frac, t0, t1
  real(8), allocatable :: x(:), y(:)

  ! Array size
  n = 102400
  call get_command_argument(1, arg)
  if (len_trim(arg) > 0) then
    read(arg, *) n
  end if
  print '(A, I0)', "Array size n = ", n

  allocate(x(n), y(n))

  ! Initializing arrays on device with data region
  !$omp target data map(to: x(1:n), alpha) map(tofrom: y(1:n))
  do i = 1, n
    frac = 1.0d0 / real(n - 1, kind=8)
    x(i) = real(i - 1, kind=8) * frac
    y(i) = real(i - 1, kind=8) * frac * 100.0d0
  end do

  !$omp target update to(x(1:n), y(1:n))

  print '(A)', "Input:"
  print '(A, F8.4)', "a = ", alpha
  call print_array("x", x)
  call print_array("y", y)

  ! Calculate axpy
  !$omp target teams distribute parallel do
  do i = 1, n
    y(i) = y(i) + alpha * x(i)
  end do
  !$omp end teams distribute parallel do

  !$omp target update from(y(1:n))
  !$omp end target data

  print '(A)', "Output:"
  call print_array("y", y)

  deallocate(x, y)

end program axpy

