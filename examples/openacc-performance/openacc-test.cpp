#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

int main( int argc, char* argv[] )
{

  struct timeval start, end;
  double time; 
	  
  int M;
  int N;

  Kokkos::initialize( argc, argv );
  {

  //for (int i = 100000; i < 100000000; i += 100000 )
  //{
  int i = 100000000;

  M = i;
  N = i;
 
  auto X  = static_cast<float*>(Kokkos::kokkos_malloc<>(M * sizeof(float)));
  auto Y  = static_cast<float*>(Kokkos::kokkos_malloc<>(M * sizeof(float)));

  typedef Kokkos::MDRangePolicy< Kokkos::Rank<2> > mdrange_policy;

  //printf("AXPY -- kokkos parallel_for\n");
  Kokkos::parallel_for( "axpy_init", M, KOKKOS_LAMBDA ( int m )
  {
    X[m] = 2.0;
    Y[m] = 2.0;
    //printf("X[%d] = %2.f and Y[%d] = %2.f\n", m, X[m], m, Y[m]);
  });

  Kokkos::fence();

  // Warming
  Kokkos::parallel_for( "axpy_comp", M, KOKKOS_LAMBDA ( int m )
  {
    float alpha = 2.0;
    Y[m] += alpha * X[m];
    //printf("Y[%d] = %2.f\n", m, Y[m]);
  });
  
  Kokkos::fence();

  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < 100; i++ )
  { 

  Kokkos::parallel_for( "axpy_comp", M, KOKKOS_LAMBDA ( int m )
  {
    float alpha = 2.0;
    Y[m] += alpha * X[m];
    //printf("Y[%d] = %2.f\n", m, Y[m]);
  });
  
  Kokkos::fence();
  }

  gettimeofday( &end, NULL );

  time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6) / 100.0;

  printf( "AXPY = %d Time ( %e s )\n", M, time );

  /*
  //printf("DOT Product -- Kokkos parallel_reduce\n");
  Kokkos::parallel_for( "dotproduct_init", M, KOKKOS_LAMBDA ( int m )
  {
    X[m] = 2.0;
    Y[m] = 2.0;
  });

  float result;

  //Warming
  Kokkos::parallel_reduce( "dotproduct_comp", N, KOKKOS_LAMBDA ( int m, float &update )
  {
    update += X[m] * Y[m];
  }, result );
 
  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < 10; i++ )
  { 

  Kokkos::parallel_reduce( "dotproduct_comp", N, KOKKOS_LAMBDA ( int m, float &update )
  {
    update += X[m] * Y[m];
  }, result );
 
  Kokkos::fence();
  }

  gettimeofday( &end, NULL );

  time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6) / 10.0;

  printf( "DOT = %d Time ( %e s )\n", M, time );
  */

  //printf("DOT Product result %2.f\n", result);
 
  /*
  printf("Kokkos parallel_for multi-dimensional\n");

  Kokkos::parallel_for( "gemv_init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    A[m * N + n] = 2.0;
    //printf("A[%d] = %2.f\n", m * N + n, A[m * N + n]);
  });

  printf("Kokkos parallel_scan input\n");
  Kokkos::parallel_for( "scan_init", M, KOKKOS_LAMBDA ( int m )
  {
    X[m] = (float)m;
    X[m] += 1.0;
    printf("X[%d] = %2.f\n", m, X[m]);
  });
  
  printf("Kokkos parallel_reduce multi-dimensional\n");
  Kokkos::parallel_for( "matrix_init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    A[m * N + n] = 2.0;
    printf("A[%d] = %2.f\n", m * N + n, A[m * N + n]);
  });

  Kokkos::parallel_reduce( "matrix_comp", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, float &update )
  {
    update += A[m * N + n];
  }, result );
  
  printf("Paralle reduce (MD) result %2.f\n", result);
 
  printf("Kokkos parallel_scan exclusive ouput\n");
  Kokkos::parallel_scan( "scan_exclusive_comp", M, KOKKOS_LAMBDA ( int m, float& update, bool final )
  {
    float tmp = X[m];
    if (final) 
    {
      X[m] = update; 
    }
    update += tmp;
    printf("X[%d] = %2.f\n", m, X[m]);
  });
  
  printf("Kokkos parallel_scan input\n");
  Kokkos::parallel_for( "scan_init", M, KOKKOS_LAMBDA ( int m )
  {
    X[m] = (float)m;
    X[m] += 1.0;
    printf("X[%d] = %2.f\n", m, X[m]);
  });

  printf("Kokkos parallel_scan inclusive ouput\n");
  Kokkos::parallel_scan( "scan_inclusive_comp", M, KOKKOS_LAMBDA ( int m, float& update1, bool final )
  {
    float tmp = X[m];
    update1 += tmp;
    if (final) 
    {
      X[m] = update1; 
    }
    printf("X[%d] = %2.f\n", m, X[m]);
  });
  */

  Kokkos::kokkos_free<>(X);
  Kokkos::kokkos_free<>(Y);

  }
  
  Kokkos::finalize();

  //}

  return 0;
}
