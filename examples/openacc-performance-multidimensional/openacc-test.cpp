#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

#define DSIZE_LB 1000
#define DSIZE_UB 10000
#define DSTEP 100
#define NITR 10
//#define LAYOUTRIGHT 1

int main( int argc, char* argv[] )
{

  struct timeval start, end;
  double time; 
	  
  int M;
  int N;

  for (int i = DSIZE_LB; i <= DSIZE_UB; i += DSTEP )
  {
  
  M = i;
  N = i;

  Kokkos::initialize( argc, argv );
  {
 
#ifdef LAYOUTRIGHT
  Kokkos::View<float **, Kokkos::LayoutRight> X("X", M, N);
  Kokkos::View<float **, Kokkos::LayoutRight> Y("Y", M, N);
#else
  Kokkos::View<float **> X("X", M, N);
  Kokkos::View<float **> Y("Y", M, N);
#endif

  typedef Kokkos::MDRangePolicy< Kokkos::Rank<2> > mdrange_policy;

  //printf("AXPY -- kokkos parallel_for\n");
  Kokkos::parallel_for( "axpy_init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    X(m,n) = 2.0;
    Y(m,n) = 2.0;
    //printf("X[%d] = %2.f and Y[%d] = %2.f\n", m, X[m], m, Y[m]);
  });

  Kokkos::fence();

  // Warming
  Kokkos::parallel_for( "axpy_comp", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    float alpha = 2.0;
    Y(m,n) += alpha * X(m,n);
    //printf("Y[%d] = %2.f\n", m, Y[m]);
  });
  
  Kokkos::fence();

  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < NITR; i++ )
  { 

  Kokkos::parallel_for( "axpy_comp", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    float alpha = 2.0;
    Y(m,n) += alpha * X(m,n);
    //printf("Y[%d] = %2.f\n", m, Y[m]);
  });
 
  Kokkos::fence();
  }

  gettimeofday( &end, NULL );

  time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6) / 100.0;

  printf( "AXPY = %d Time ( %e s )\n", M, time );

  //printf("DOT Product -- Kokkos parallel_reduce\n");
  Kokkos::parallel_for( "dotproduct_init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    X(m,n) = 2.0;
    Y(m,n) = 2.0;
    //printf("X[%d] = %2.f and Y[%d] = %2.f\n", m, X[m], m, Y[m]);
  });

  float result;

  //Warming
  Kokkos::parallel_reduce( "dotproduct_comp", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, float &update )
  {
    update += X(m,n) * Y(m,n);
  }, result );
 
  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < NITR; i++ )
  { 
  Kokkos::parallel_reduce( "dotproduct_comp", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, float &update )
  {
    update += X(m,n) * Y(m,n);
  }, result );
 
  Kokkos::fence();
  }

  gettimeofday( &end, NULL );

  time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6) / 100.0;

  printf( "DOT = %d Time ( %e s )\n", M, time );


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

  }
  

  }
  Kokkos::finalize();

  return 0;
}
