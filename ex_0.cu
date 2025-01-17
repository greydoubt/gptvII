/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//! Solves heat equation in 2D, see the README.

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <vector>

// Problem parameters
struct parameters {
  double dx, dt;
  long nx, ny, ni;
  int rank = 0, nranks = 1;

  static constexpr double alpha() { return 1.0; } // Thermal diffusivity

  parameters(int argc, char *argv[]);
    
  long nit() { return ni; }
  long nout() { return 1000; }
  long nx_global() { return nx * nranks; }
  long ny_global() { return ny; }
  double gamma() { return alpha() * dt / (dx * dx); }
  long n() { return ny * (nx + 2 /* 2 halo layers */); }
};

double stencil(double* u_new, double* u_old, long x, long y, parameters p);

// 2D grid of indicies
struct grid {
  long x_begin, x_end, y_begin, y_end;
};

double apply_stencil(double* u_new, double* u_old, grid g, parameters p) {
  double energy = 0.;
  for (long x = g.x_begin; x < g.x_end; ++x) {
    for (long y = g.y_begin; y < g.y_end; ++y) {
      energy += stencil(u_new, u_old, x, y, p);
    }
  }
  return energy;
}

// Initial condition
void initial_condition(double* u_new, double* u_old, long n) {
  for (long i = 0; i < n; ++i) {
    u_old[i] = 0.;
    u_new[i] = 0.;
  }
}

// These evolve the solution of different parts of the local domain.
double inner(double* u_new, double *u_old, parameters p);
double prev (double* u_new, double *u_old, parameters p); 
double next (double* u_new, double *u_old, parameters p);

int main(int argc, char *argv[]) {
  // Parse CLI parameters
  parameters p(argc, argv);

  // Initialize MPI with multi-threading support
  int mt;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mt);
  if (mt != MPI_THREAD_MULTIPLE) {
    std::cerr << "MPI cannot be called from multiple host threads" << std::endl;
    std::terminate();
  }
  MPI_Comm_size(MPI_COMM_WORLD, &p.nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &p.rank);

  // Allocate memory
  std::vector<double> u_new(p.n()), u_old(p.n());
 
  // Initial condition
  initial_condition(u_new.data(), u_old.data(), p.n());

  // Time loop
  using clk_t = std::chrono::steady_clock;
  auto start = clk_t::now();

  for (long it = 0; it < p.nit(); ++it) {
    // Evolve the solution:
    double energy = 
        prev(u_new.data(), u_old.data(), p) +
        next(u_new.data(), u_old.data(), p) +
        inner(u_new.data(), u_old.data(), p);

    // Reduce the energy across all neighbors to the rank == 0, and print it if necessary:
    MPI_Reduce(p.rank == 0 ? MPI_IN_PLACE : &energy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (p.rank == 0 && it % p.nout() == 0) {
      std::cerr << "E(t=" << it * p.dt << ") = " << energy << std::endl;
    }
    std::swap(u_new, u_old);
  }

  auto time = std::chrono::duration<double>(clk_t::now() - start).count();
  auto grid_size = static_cast<double>(p.nx * p.ny * sizeof(double) * 2) * 1e-9; // GB
  auto memory_bw = grid_size * static_cast<double>(p.nit()) / time;             // GB/s
  if (p.rank == 0) {
    std::cerr << "Rank " << p.rank << ": local domain " << p.nx << "x" << p.ny << " (" << grid_size << " GB): " 
              << memory_bw << " GB/s" << std::endl;
    std::cerr << "All ranks: global domain " << p.nx_global() << "x" << p.ny_global() << " (" << (grid_size * p.nranks) << " GB): " 
              << memory_bw * p.nranks << " GB/s" << std::endl; 
  }

  // Write output to file
  MPI_File f;
  MPI_File_open(MPI_COMM_WORLD, "output", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);
  auto header_bytes = 2 * sizeof(long) + sizeof(double);
  auto values_per_rank = p.nx * p.ny;
  auto values_bytes_per_rank = values_per_rank * sizeof(double);
  MPI_File_set_size(f, header_bytes + values_bytes_per_rank * p.nranks);
  MPI_Request req[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  if (p.rank == 0) {
    long total[2] = {p.nx * p.nranks, p.ny};
    double time = p.nit() * p.dt;
    MPI_File_iwrite_at(f, 0, total, 2, MPI_UINT64_T, &req[1]);
    MPI_File_iwrite_at(f, 2 * sizeof(long), &time, 1, MPI_DOUBLE, &req[2]);
  }
  auto values_offset = header_bytes + p.rank * values_bytes_per_rank;
  MPI_File_iwrite_at(f, values_offset, u_new.data() + p.ny, values_per_rank, MPI_DOUBLE, &req[0]);
  MPI_Waitall(p.rank == 0 ? 3 : 1, req, MPI_STATUSES_IGNORE);
  MPI_File_close(&f);

  MPI_Finalize();
  return 0;
}

// Reads command line arguments to initialize problem size
parameters::parameters(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "ERROR: incorrect arguments" << std::endl;
    std::cerr << "  " << argv[0] << " <nx> <ny> <ni>" << std::endl;
    std::terminate();
  }
  nx = std::stoll(argv[1]);
  ny = std::stoll(argv[2]);
  ni = std::stoll(argv[3]);
  dx = 1.0 / nx;
  dt = dx * dx / (5. * alpha());
}

// Finite-difference stencil
double stencil(double *u_new, double *u_old, long x, long y, parameters p) {
  auto idx = [=](auto x, auto y) { 
      // Index into the memory using row-major order:
      assert(x >= 0 && x < 2 * p.nx);
      assert(y >= 0 && y < p.ny);
      return x * p.ny + y;
  };
  // Apply boundary conditions:
  if (y == 1) {
    u_old[idx(x, y - 1)] = 0;
  }
  if (y == (p.ny - 2)) {
    u_old[idx(x, y + 1)] = 0;
  }
  // These boundary conditions are only impossed by the ranks at the end of the domain:
  if (p.rank == 0 && x == 1) {
    u_old[idx(x - 1, y)] = 1;
  }
  if (p.rank == (p.nranks - 1) && x == p.nx) {
    u_old[idx(x + 1, y)] = 0;
  }

  u_new[idx(x, y)] = (1. - 4. * p.gamma()) * u_old[idx(x, y)] +
                     p.gamma() * (u_old[idx(x + 1, y)] + u_old[idx(x - 1, y)] +
                                  u_old[idx(x, y + 1)] + u_old[idx(x, y - 1)]);

  return u_new[idx(x, y)] * p.dx * p.dx;
}

// Evolve the solution of the interior part of the domain
// which does not depend on data from neighboring ranks
double inner(double *u_new, double *u_old, parameters p) {
  grid g{.x_begin = 2, .x_end = p.nx, .y_begin = 1, .y_end = p.ny - 1};
  return apply_stencil(u_new, u_old, g, p);
}

// Evolve the solution of the part of the domain that 
// depends on data from the previous MPI rank (rank - 1)
double prev(double *u_new, double *u_old, parameters p) {
  // Send window cells, receive halo cells
  if (p.rank > 0) {
    // Send bottom boundary to bottom rank
    MPI_Send(u_old + p.ny, p.ny, MPI_DOUBLE, p.rank - 1, 0, MPI_COMM_WORLD);
    // Receive top boundary from bottom rank
    MPI_Recv(u_old + 0, p.ny, MPI_DOUBLE, p.rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  // Compute prev boundary
  grid g{.x_begin = 1, .x_end = 2, .y_begin = 1, .y_end = p.ny - 1};
  return apply_stencil(u_new, u_old, g, p);
}

// Evolve the solution of the part of the domain that 
// depends on data from the next MPI rank (rank + 1)
double next(double *u_new, double *u_old, parameters p) {
  if (p.rank < p.nranks - 1) {
    // Receive bottom boundary from top rank
    MPI_Recv(u_old + (p.nx + 1) * p.ny, p.ny, MPI_DOUBLE, p.rank + 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    // Send top boundary to top rank, and
    MPI_Send(u_old + p.nx * p.ny, p.ny, MPI_DOUBLE, p.rank + 1, 1, MPI_COMM_WORLD);
  }
  // Compute next boundary
  grid g{.x_begin = p.nx, .x_end = p.nx + 1, .y_begin = 1, .y_end = p.ny - 1};
  return apply_stencil(u_new, u_old, g, p);
}
