#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

int main( int argc, char* argv[] )
{
  int M = 4;
  int N = 4;

  Kokkos::initialize( argc, argv );
  {
 
  auto X   = static_cast<float*>(Kokkos::kokkos_malloc<>(M * N * sizeof(float)));
  auto IX  = static_cast<int*>(Kokkos::kokkos_malloc<>(M * N * sizeof(int)));
  
  float result;
  int iresult;
  
  typedef Kokkos::MDRangePolicy< Kokkos::Rank<2> > mdrange_policy;

  // --------------Sum--------------
  
  Kokkos::parallel_for( "init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    X[m * N + n] = 2.5;
    printf("X[%d] = %2.1f\n", m * N + n, X[m * N + n]);
  });

  Kokkos::fence();

  Kokkos::parallel_reduce( "Sum reduction", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, float &update )
  {
    update += X[m * N + n];
  }, Kokkos::Sum<float>(result) );
  
  Kokkos::fence();
  
  printf("Sum result (float) %2.1f\n", result);

  // --------------Prod--------------
 
  Kokkos::parallel_for( "init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    X[m * N + n] = 2.5;
    printf("X[%d] = %2.1f\n", m * N + n, X[m * N + n]);
  });

  Kokkos::fence();
  
  Kokkos::parallel_reduce( "Prod reduction", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, float &update )
  {
    update *= X[m * N + n];
  }, Kokkos::Prod<float>(result) );
  
  Kokkos::fence();
  
  printf("Prod result (float) %2.1f\n", result);

  // --------------Min--------------
  
  result = 0.0;
  
  Kokkos::parallel_for( "init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    X[m * N + n] = 2.0 + (float) ( m * N + n );
    printf("X[%d] = %2.1f\n", m * M + n, X[m * N + n]);
  });

  Kokkos::fence();
  
  Kokkos::parallel_reduce( "Min reduction", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, float &update )
  {
    update = X[0];
    float a = X[m * N + n];
    update = std::min( a, update );
  }, Kokkos::Min<float>(result) );
  
  Kokkos::fence();
  
  printf("Min result %2.1f\n", result);
 
  // --------------Max--------------

  result = 0.0;

  Kokkos::parallel_for( "init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    X[m * N + n] = 2.5 + (float) ( m * N + n );
    printf("X[%d] = %2.1f\n", m * N + n, X[m * N + n]);
  });

  Kokkos::fence();
  
  Kokkos::parallel_reduce( "Max reduction", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, float &update )
  {
    update = X[0];
    float a = X[m * N + n];
    update = std::max( a, update );
  }, Kokkos::Max<float>(result) );
  
  Kokkos::fence();
  
  printf("Max result %2.1f\n", result);

  // --------------LAnd--------------

  Kokkos::parallel_for( "init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    if ((m * N + n) % 2 == 0)
    {
      IX[m * N + n] = 1;
    }
    else
    {
      IX[m * N + n] = 0;
    }
    printf("IX[%d] = %s\n", m * N + n, IX[m * N + n] ? "true" : "false");
  });

  Kokkos::fence();

  Kokkos::parallel_reduce( "Land reduction", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, int &iupdate )
  {
    iupdate = iupdate && IX[m * N + n];
  }, Kokkos::LAnd<int>(iresult) );
  
  Kokkos::fence();
  
  printf("LAnd result %s\n", iresult ? "true" : "false");

  // --------------LOr--------------

  Kokkos::parallel_for( "init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    if ((m * N + n) % 2 == 0)
    {
      IX[m * N + n] = 1;
    }
    else
    {
      IX[m * N + n] = 0;
    }
    printf("IX[%d] = %s\n", m * N + n, IX[m * N + n] ? "true" : "false");
  });

  Kokkos::fence();

  Kokkos::parallel_reduce( "LOr reduction", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, int &iupdate )
  {
    iupdate = iupdate || IX[m * N + n];
  }, Kokkos::LOr<int>(iresult) );
  
  Kokkos::fence();
  
  printf("LOr result %s\n", iresult ? "true" : "false");

  // --------------BAnd--------------

  Kokkos::parallel_for( "init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    IX[m * N + n] = 1;
    printf("IX[%d] = %d\n", m * N + n, IX[m * N + n]);
  });

  Kokkos::fence();
 
  Kokkos::parallel_reduce( "BAnd reduction", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, int &iupdate )
  {
    iupdate = iupdate & IX[m * N + n];
  }, Kokkos::BAnd<int>(iresult) );
  
  printf("BAnd result %d\n", iresult);

  // --------------BOr--------------

  Kokkos::parallel_for( "init", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n )
  {
    IX[m * N + n] = 2 + (m * N + n);
    printf("IX[%d] = %d\n", m * N + n, IX[m * N + n]);
  });

  Kokkos::fence();
  
  Kokkos::parallel_reduce( "BOr reduction", mdrange_policy( {0, 0}, {M, N}), KOKKOS_LAMBDA ( int m, int n, int &iupdate )
  {
    iupdate = iupdate | IX[m * N + n];
  }, Kokkos::BOr<int>(iresult) );
  
  Kokkos::fence();
  
  printf("BOr result %d\n", iresult);

  Kokkos::kokkos_free<>(X);
  Kokkos::kokkos_free<>(IX);

  }
  
  Kokkos::finalize();

  return 0;
}
