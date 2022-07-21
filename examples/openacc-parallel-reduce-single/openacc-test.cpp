#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

int main( int argc, char* argv[] )
{
  int M = 4;

  Kokkos::initialize( argc, argv );
  {
 
  auto X   = static_cast<float*>(Kokkos::kokkos_malloc<>(M * sizeof(float)));
  auto IX  = static_cast<int*>(Kokkos::kokkos_malloc<>(M * sizeof(int)));
  
  // --------------Sum--------------

  float result;
  int iresult;

  Kokkos::parallel_for( "init", M, KOKKOS_LAMBDA ( int m )
  {
    X[m] = 2.5;
    printf("X[%d] = %2.1f\n", m, X[m]);
  });

  Kokkos::fence();

  Kokkos::parallel_reduce( "Sum reduction", M, KOKKOS_LAMBDA ( int m, float &update )
  {
    update += X[m];
  }, Kokkos::Sum<float>(result) );
  
  Kokkos::fence();
  
  printf("Sum result (float) %2.1f\n", result);

  Kokkos::parallel_for( "init", M, KOKKOS_LAMBDA ( int m )
  {
    IX[m] = 2.5;
    printf("X[%d] = %d\n", m, IX[m]);
  });

  Kokkos::fence();

  Kokkos::parallel_reduce( "Sum reduction", M, KOKKOS_LAMBDA ( int m, int &iupdate )
  {
    iupdate += IX[m];
  }, Kokkos::Sum<int>(iresult) );
  
  Kokkos::fence();
  
  printf("Sum result (int) %d\n", iresult);


  // --------------Prod--------------
 
  Kokkos::parallel_for( "init", M, KOKKOS_LAMBDA ( int m )
  {
    X[m] = 2.5;
    printf("X[%d] = %2.1f\n", m, X[m]);
  });

  Kokkos::fence();
  
  Kokkos::parallel_reduce( "Prod reduction", M, KOKKOS_LAMBDA ( int m, float &update )
  {
    update *= X[m];
  }, Kokkos::Prod<float>(result) );
  
  Kokkos::fence();
  
  printf("Prod result (float) %2.1f\n", result);

  Kokkos::parallel_for( "init", M, KOKKOS_LAMBDA ( int m )
  {
    IX[m] = 2;
    printf("X[%d] = %d\n", m, IX[m]);
  });

  Kokkos::fence();
  
  Kokkos::parallel_reduce( "Prod reduction", M, KOKKOS_LAMBDA ( int m, int &iupdate )
  {
    iupdate *= IX[m];
  }, Kokkos::Prod<int>(iresult) );
  
  Kokkos::fence();
  
  printf("Prod result (int) %d\n", iresult);

  // --------------Min--------------
  
  result = 0.0;
  
  Kokkos::parallel_for( "init", M, KOKKOS_LAMBDA ( int m )
  {
    X[m] = 2.0 + (float) m;
    printf("X[%d] = %2.1f\n", m, X[m]);
  });

  Kokkos::fence();
  
  Kokkos::parallel_reduce( "Min reduction", M, KOKKOS_LAMBDA ( int m, float &update )
  {
    update = X[0];
    float a = X[m];
    update = std::min( a, update );
  }, Kokkos::Min<float>(result) );
  
  Kokkos::fence();
  
  printf("Min result %2.1f\n", result);
 
  // --------------Max--------------

  result = 0.0;

  Kokkos::parallel_for( "init", M, KOKKOS_LAMBDA ( int m )
  {
    X[m] = 2.5 + (float) m;
    printf("X[%d] = %2.1f\n", m, X[m]);
  });

  Kokkos::fence();
  
  Kokkos::parallel_reduce( "Max reduction", M, KOKKOS_LAMBDA ( int m, float &update )
  {
    update = X[0];
    float a = X[m];
    update = std::max( a, update );
  }, Kokkos::Max<float>(result) );
  
  Kokkos::fence();
  
  printf("Max result %2.1f\n", result);

  // --------------LAnd--------------

  Kokkos::parallel_for( "init", M, KOKKOS_LAMBDA ( int m )
  {
    if (m % 2 == 0)
    {
      IX[m] = 0;
    }
    else
    {
      IX[m] = 1;
    }
    printf("IX[%d] = %s\n", m, IX[m] ? "true" : "false");
  });

  Kokkos::fence();

  Kokkos::parallel_reduce( "Land reduction", M, KOKKOS_LAMBDA ( int m, int &iupdate )
  {
    iupdate = iupdate && IX[m];
  }, Kokkos::LAnd<int>(iresult) );
  
  Kokkos::fence();
  
  printf("LAnd result %s\n", iresult ? "true" : "false");

  // --------------LOr--------------

  Kokkos::parallel_for( "init", M, KOKKOS_LAMBDA ( int m )
  {
    if (m % 2 == 0)
    {
      IX[m] = 0;
    }
    else
    {
      IX[m] = 1;
    }
    printf("IX[%d] = %s\n", m, IX[m] ? "true" : "false");
  });

  Kokkos::fence();

  Kokkos::parallel_reduce( "LOr reduction", M, KOKKOS_LAMBDA ( int m, int &iupdate )
  {
    iupdate = iupdate || IX[m];
  }, Kokkos::LOr<int>(iresult) );
  
  Kokkos::fence();
  
  printf("LOr result %s\n", iresult ? "true" : "false");

  // --------------BAnd--------------

  Kokkos::parallel_for( "init", M, KOKKOS_LAMBDA ( int m )
  {
    IX[m] = 1;
    printf("IX[%d] = %d\n", m, IX[m]);
  });

  Kokkos::fence();
 
  Kokkos::parallel_reduce( "BAnd reduction", M, KOKKOS_LAMBDA ( int m, int &iupdate )
  {
    iupdate = iupdate & IX[m];
  }, Kokkos::BAnd<int>(iresult) );
  
  printf("BAnd result %d\n", iresult);

  // --------------BOr--------------

  Kokkos::parallel_for( "init", M, KOKKOS_LAMBDA ( int m )
  {
    IX[m] = 2 + m;
    printf("IX[%d] = %d\n", m, IX[m]);
  });

  Kokkos::fence();
  
  Kokkos::parallel_reduce( "BOr reduction", M, KOKKOS_LAMBDA ( int m, int &iupdate )
  {
    iupdate = iupdate | IX[m];
  }, Kokkos::BOr<int>(iresult) );
  
  Kokkos::fence();
  
  printf("BOr result %d\n", iresult);

  Kokkos::kokkos_free<>(X);
  Kokkos::kokkos_free<>(IX);

  }
  
  Kokkos::finalize();

  return 0;
}
