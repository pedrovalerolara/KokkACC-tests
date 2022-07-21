#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>
//#define LAYOUTRIGHT 1

int main( int argc, char* argv[] )
{

  struct timeval start, end;
  double time; 
	  
  int M;
  int N;

  Kokkos::initialize( argc, argv );
  {
  
  typedef Kokkos::TeamPolicy<>               team_policy;
  typedef Kokkos::TeamPolicy<>::member_type  member_type;
  
  for (int i = 1000; i < 10000; i += 100 )
  {
  
  M = i;
  N = i;

 
#ifdef LAYOUTRIGHT
  Kokkos::View<float **, Kokkos::LayoutRight> X("X", M, N); 
  Kokkos::View<float **, Kokkos::LayoutRight> Y("Y", M, N); 
#else
  Kokkos::View<float **> X("X", M, N); 
  Kokkos::View<float **> Y("Y", M, N); 
#endif

  typedef Kokkos::MDRangePolicy< Kokkos::Rank<2> > mdrange_policy;

  Kokkos::parallel_for( "axpy_init", team_policy( M, Kokkos::AUTO ), KOKKOS_LAMBDA ( const member_type &teamMember )
  {
    const int m = teamMember.league_rank();

    Kokkos::parallel_for( Kokkos::TeamThreadRange( teamMember, N ), [&] ( const int n )
    {
      X(n,m) = 2.0;
      Y(n,m) = 2.0;
    });
  });
  
  Kokkos::fence();

  // Warming
  Kokkos::parallel_for( "axpy_comp", team_policy( M, Kokkos::AUTO ), KOKKOS_LAMBDA ( const member_type &teamMember )
  {
    const int m = teamMember.league_rank();

    Kokkos::parallel_for( Kokkos::TeamThreadRange( teamMember, N ), [&] ( const int n )
    {
      float alpha = 2.0;
      Y(n,m) += alpha * X(n,m);
    });
  });
  
  Kokkos::fence();

  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < 10; i++ )
  { 
  Kokkos::parallel_for( "axpy_comp", team_policy( M, Kokkos::AUTO ), KOKKOS_LAMBDA ( const member_type &teamMember )
  {
    const int m = teamMember.league_rank();

    Kokkos::parallel_for( Kokkos::TeamThreadRange( teamMember, N ), [&] ( const int n )
    {
      float alpha = 2.0;
      Y(n,m) += alpha * X(n,m);
    });
  });
 
  Kokkos::fence();
  }

  gettimeofday( &end, NULL );

  time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec)
                                 - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6) / 100.0;

  printf( "AXPY = %d Time ( %e s )\n", M, time );

  Kokkos::parallel_for( "dotproduct_init", team_policy( M, Kokkos::AUTO ), KOKKOS_LAMBDA ( const member_type &teamMember )
  {
    const int m = teamMember.league_rank();

    Kokkos::parallel_for( Kokkos::TeamThreadRange( teamMember, N ), [&] ( const int n )
    {
      X(n,m) = 2.0;
      Y(n,m) = 2.0;
    });
  });
  
  Kokkos::fence();

  float result = 0.0;

  //Warming
  Kokkos::parallel_reduce( "dotproduct_comp", team_policy( M, Kokkos::AUTO ), KOKKOS_LAMBDA ( const member_type &teamMember, float &update )
  {
    const int m = teamMember.league_rank();
    float tmp = 0.0;

    Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, N ), [&] ( const int n, float &innerUpdate ) {
      innerUpdate += X(n,m) * Y(n,m);
    }, tmp );

    //printf("tmp %f\n", tmp);
    if ( teamMember.team_rank() == 0 ) update += tmp;

  }, result );
  //printf("result %f\n", result);
  Kokkos::fence();

  //Time
  gettimeofday( &start, NULL );
  for ( int i = 0; i < 100; i++ )
  {
  Kokkos::parallel_reduce( "dotproduct_comp", team_policy( M, Kokkos::AUTO ), KOKKOS_LAMBDA ( const member_type &teamMember, float &update )
  {
    const int m = teamMember.league_rank();
    float tmp = 0.0;

    Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, N ), [&] ( const int n, float &innerUpdate ) {
      innerUpdate += X(n,m) * Y(n,m);
    }, tmp );

    if ( teamMember.team_rank() == 0 ) update += tmp;

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
  
  Kokkos::finalize();

  }

  return 0;
}
