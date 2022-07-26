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

  Kokkos::initialize( argc, argv );
  {

  //for (int i = 100000; i < 100000000; i += 100000 )
  for (int i = 100000; i <= 100000; i += 100000 )
  {

  M = i;
 
  auto X  = static_cast<double*>(Kokkos::kokkos_malloc<>(M * sizeof(double)));
  auto Y  = static_cast<double*>(Kokkos::kokkos_malloc<>(M * sizeof(double)));

  typedef Kokkos::MDRangePolicy< Kokkos::Rank<2> > mdrange_policy;

  Kokkos::parallel_for( "scan_exclusive_init", M, KOKKOS_LAMBDA ( int m )
  {
    X[m] = (double)m;
    X[m] += 1.0;
    //printf("Input X[%d] = %e\n", m, X[m]);
  });

  Kokkos::fence();

  // Warming
  Kokkos::parallel_scan( "scan_exclusive_comp", M, KOKKOS_LAMBDA ( int m, float& update, bool final )
  {
    float tmp = X[m];
    if (final) 
    {
      X[m] = update; 
    }
    update += tmp;
    //printf("Scan processing X[%d] = %e\n", m, X[m]);
  });
  
  Kokkos::fence();

  /*
  Kokkos::parallel_for( "scan_exclusive_output", M, KOKKOS_LAMBDA ( int m )
  {
    printf("Output X[%d] = %e\n", m, X[m]);
  });

  Kokkos::fence();
  */

  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < 10; i++ )
  { 
  Kokkos::parallel_scan( "scan_exclusive_comp", M, KOKKOS_LAMBDA ( int m, float& update, bool final )
  {
    float tmp = X[m];
    if (final) 
    {
      X[m] = update; 
    }
    update += tmp;
  });
 
  Kokkos::fence();
  }

  gettimeofday( &end, NULL );

  time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6) / 10.0;

  printf( "SCAN EXCLUSIVE = %d Time ( %e s )\n", M, time );

  /*
  Kokkos::parallel_for( "scan_inclusive_init", M, KOKKOS_LAMBDA ( int m )
  {
    Y[m] = 2.0;
  });

  float result;

  //Warming
  Kokkos::parallel_scan( "scan_inclusive_comp", M, KOKKOS_LAMBDA ( int m, float& update1, bool final )
  {
    float tmp = Y[m];
    update1 += tmp;
    if (final) 
    {
      Y[m] = update1; 
    }
    //printf("X[%d] = %2.f\n", m, Y[m]);
  });

  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < 10; i++ )
  { 

  Kokkos::parallel_scan( "scan_inclusive_comp", M, KOKKOS_LAMBDA ( int m, float& update1, bool final )
  {
    float tmp = Y[m];
    update1 += tmp;
    if (final) 
    {
      Y[m] = update1; 
    }
    //printf("X[%d] = %2.f\n", m, Y[m]);
  });

  Kokkos::fence();
  }

  gettimeofday( &end, NULL );

  time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6) / 10.0;

  printf( "SCAN INCLUSIVE = %d Time ( %e s )\n", M, time );
  */

  Kokkos::kokkos_free<>(X);
  Kokkos::kokkos_free<>(Y);

  }
  
  Kokkos::finalize();

  }

  return 0;
}
