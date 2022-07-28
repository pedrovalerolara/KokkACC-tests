#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>


#define M (8)

#if 0
template <class T>
//T testAtomic(std::string& Name, std::string& Type, T initial, T expected, T* Data, T (*atomic)(volatile T*, const T&))
T testAtomic(std::string Name, T initial, T expected, T* Data, void *atomic)
{
  auto F = reinterpret_cast<T (*)(volatile T*, const T)>(atomic);

  printf("Test for %s -- kokkos parallel_for\n", Name);
  Kokkos::parallel_for( "Init", M, KOKKOS_LAMBDA ( int m )
  {
    Data[m] = initial;
  });
  Kokkos::fence();

  Kokkos::parallel_for( "Testing atomic", M, KOKKOS_LAMBDA ( int m )
  {
    F(&Data[0], 1);
  });
  Kokkos::fence();

  Kokkos::parallel_for( "Verify atomic", 1, KOKKOS_LAMBDA ( int m )
  {
    if (Data[0]!=expected) printf("FAILED: %s  with type (%s): (%i) output, expected (%i).\n", Name, Type, Data[0], expected);
    else printf("VERIFIED: %s  with type (%s): (%i) output, expected (%i).\n", Name, Type, Data[0], expected);
  });
  Kokkos::fence();
}
#endif

#define verifyAtomic(Name, Type, initial, expected, OP, Data, atomic)\
{\
  Kokkos::parallel_for( "Verify atomic", 1, KOKKOS_LAMBDA ( int m )\
  {\
    Type A=static_cast<Type>(expected);\
    if (sizeof(Type)<=4) {\
      if (Data[0]!=A)\
        printf("        FAILED: %s  with type (%s),  output = (0x%.8x), expected (0x%.8x)\n", #Name, #Type, Data[0], A);\
      else\
        printf("        VERIFIED: %s  with type (%s),  output = (0x%.8x), expected (0x%.8x)\n", #Name, #Type, Data[0], A);\
    }\
    else {\
      if (Data[0]!=A) \
        printf("        FAILED: %s  with type (%s),  output = (0x%.16x), expected (0x%.16x)\n", #Name, #Type, Data[0], A);\
      else \
        printf("        VERIFIED: %s  with type (%s),  output = (0x%.16x), expected (0x%.16x)\n", #Name, #Type, Data[0], A);\
    }\
  });\
  Kokkos::fence();\
}

#define callAtomic2(Name, Type, initial, expected, OP, Data, atomic)\
{\
  Kokkos::parallel_for( "Testing atomic", M, KOKKOS_LAMBDA ( int m )\
  {\
    Type A=static_cast<Type>(OP);\
/*printf("before Data[0]: (0x%.16x) OP: (0x%.16x)\n", Data[0], A);*/\
    atomic(&Data[0], A);\
/*if (sizeof(Type)<=4) \
printf("after Data[0]: (0x%.8x) OP: (0x%.8x)\n", Data[0], A);\
else \
printf("after Data[0]: (0x%.16x) OP: (0x%.16x)\n", Data[0], A); */\
  });\
  Kokkos::fence();\
}

#define callAtomic3(Name, Type, initial, expected, OP1, OP2, Data, atomic)\
{\
  Kokkos::parallel_for( "Testing atomic", M, KOKKOS_LAMBDA ( int m )\
  {\
    Type A=static_cast<Type>(OP1);\
    Type B=static_cast<Type>(OP2);\
/*printf("before Data[0]: (0x%.16x) OP: (0x%.16x)\n", Data[0], A);*/\
    atomic(&Data[0], A, B);\
/*if (sizeof(Type)<=4) \
printf("after Data[0]: (0x%.8x) OP: (0x%.8x)\n", Data[0], A);\
else \
printf("after Data[0]: (0x%.16x) OP: (0x%.16x)\n", Data[0], A); */\
  });\
  Kokkos::fence();\
}


#define initAtomic(Name, Type, initial, expected, OP, Data, atomic)\
{\
  printf("\n\n ------------- Test for %s with type %s ----------- \n", #Name, #Type);\
  Kokkos::parallel_for( "Init", M, KOKKOS_LAMBDA ( int m )\
  {\
    Type A=static_cast<Type>(initial);\
    Data[m] = A;\
/*printf("Data[m]: (0x%.16x) Initial: (0x%.16x) A: (0x%.16x)\n", Data[m], (Type)initial, A);*/\
  });\
  Kokkos::fence();\
}

#define testAtomic2(Name, Type, initial, expected, OP, Data, atomic)\
{\
	initAtomic(Name, Type, initial, expected, OP, Data, atomic);\
	callAtomic2(Name, Type, initial, expected, OP, Data, atomic);\
	verifyAtomic(Name, Type, initial, expected, OP, Data, atomic)\
}

#define testAtomic3(Name, Type, initial, expected, OP1, OP2, Data, atomic)\
{\
        initAtomic(Name, Type, initial, expected, OP1, Data, atomic);\
        callAtomic3(Name, Type, initial, expected, OP1, OP2, Data, atomic);\
        verifyAtomic(Name, Type, initial, expected, OP1, Data, atomic)\
}

#if 0
#define testAtomic2(Name, Type, initial, expected, OP, Data, atomic)\
{\
  printf("\n\n ------------- Test for %s with type %s ----------- \n", #Name, #Type);\
  Kokkos::parallel_for( "Init", M, KOKKOS_LAMBDA ( int m )\
  {\
    Type A=static_cast<Type>(initial);\
    Data[m] = A;\
/*printf("Data[m]: (0x%.16x) Initial: (0x%.16x) A: (0x%.16x)\n", Data[m], (Type)initial, A);*/\
  });\
  Kokkos::fence();\
\
  Kokkos::parallel_for( "Testing atomic", M, KOKKOS_LAMBDA ( int m )\
  {\
    Type A=static_cast<Type>(OP);\
/*printf("before Data[0]: (0x%.16x) OP: (0x%.16x)\n", Data[0], A);*/\
    atomic(&Data[0], A);\
/*if (sizeof(Type)<=4) \
printf("after Data[0]: (0x%.8x) OP: (0x%.8x)\n", Data[0], A);\
else \
printf("after Data[0]: (0x%.16x) OP: (0x%.16x)\n", Data[0], A); */\
  });\
  Kokkos::fence();\
\
  Kokkos::parallel_for( "Verify atomic", 1, KOKKOS_LAMBDA ( int m )\
  {\
    Type A=static_cast<Type>(expected);\
    if (sizeof(Type)<=4) {\
      if (Data[0]!=A)\
        printf("        FAILED: %s  with type (%s),  output = (0x%.8x), expected (0x%.8x)\n", #Name, #Type, Data[0], A);\
      else\
        printf("        VERIFIED: %s  with type (%s),  output = (0x%.8x), expected (0x%.8x)\n", #Name, #Type, Data[0], A);\
    }\
    else {\
      if (Data[0]!=A) \
        printf("        FAILED: %s  with type (%s),  output = (0x%.16x), expected (0x%.16x)\n", #Name, #Type, Data[0], A);\
      else \
        printf("        VERIFIED: %s  with type (%s),  output = (0x%.16x), expected (0x%.16x)\n", #Name, #Type, Data[0], A);\
    }\
  });\
  Kokkos::fence();\
}
#endif

Kokkos::Timer kokkACCTimer;
const char *RegionName;

void resetTimer(const char*Name, const uint32_t A, uint64_t* B) 
{
    kokkACCTimer.reset();
    RegionName = Name;
}
void readTimer(uint64_t B) 
{
    double time=0.0;
    time += kokkACCTimer.seconds();
    std::cerr << "Timer for execution for REGION (" << RegionName << "): " << time << std::endl;
}


int main( int argc, char* argv[] )
{

  Kokkos::initialize( argc, argv );

  {
 
  auto Vchar  = static_cast<char*>(Kokkos::kokkos_malloc<>(M * sizeof(char)));
  auto Vshort  = static_cast<short*>(Kokkos::kokkos_malloc<>(M * sizeof(short)));
  auto Vfloat  = static_cast<float*>(Kokkos::kokkos_malloc<>(M * sizeof(float)));
  auto Vuint  = static_cast<unsigned int*>(Kokkos::kokkos_malloc<>(M * sizeof(unsigned int)));
  auto Vint  = static_cast<int*>(Kokkos::kokkos_malloc<>(M * sizeof(int)));
  auto Vull  = static_cast<unsigned long long int*>(Kokkos::kokkos_malloc<>(M * sizeof(unsigned long long int)));
  auto Vll  = static_cast<long long int*>(Kokkos::kokkos_malloc<>(M * sizeof(long long int)));

  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(resetTimer);
  Kokkos::Tools::Experimental::set_end_parallel_for_callback(readTimer);

  unsigned int Expected=4;
  unsigned int Initial=0;
  unsigned int Operand=4;
  //testAtomic2(atomic_exchange, char, char(Initial), char(Expected), char(Operand), Vchar, Kokkos::atomic_exchange);
  //testAtomic2(atomic_exchange, short, short(Initial), short(Expected),  short(Operand), Vshort, Kokkos::atomic_exchange);
  testAtomic2(atomic_exchange, float, float(Initial), float(Expected), float(Operand), Vfloat, Kokkos::atomic_exchange);
  testAtomic2(atomic_exchange, unsigned int, Initial, Expected, Operand, Vuint, Kokkos::atomic_exchange);
  testAtomic2(atomic_exchange, int, Initial, Expected, Operand, Vint, Kokkos::atomic_exchange);
  testAtomic2(atomic_exchange, unsigned long long int, Initial, Expected, Operand, Vull, Kokkos::atomic_exchange);

  Expected=4;
  Initial=4;
  Operand=0;
  int Compare=4;
  //testAtomic3(atomic_compare_exchange, short, Initial, Expected, Operand, Compare, Vshort, Kokkos::atomic_compare_exchange);
  testAtomic3(atomic_compare_exchange, unsigned int, Initial, Expected, Operand, Compare, Vuint, Kokkos::atomic_compare_exchange);
  testAtomic3(atomic_compare_exchange, int, Initial, Expected, Operand, Compare, Vint, Kokkos::atomic_compare_exchange);
  testAtomic3(atomic_compare_exchange, unsigned long long int, Initial, Expected, Operand, Compare, Vull, Kokkos::atomic_compare_exchange);
  //testAtomic3(atomic_compare_exchange, float, Initial, Expected, Operand, Compare, Vfloat, Kokkos::atomic_compare_exchange);

  Expected=M;
  Initial=0;
  Operand = 1;
  std::cout << "\n\nChecking atomic_fetch_add: " << std::endl;
  //testAtomic2(atomic_fetch_add, char, Initial, Expected, Operand, Vchar, Kokkos::atomic_fetch_add);
  //testAtomic2(atomic_fetch_add, short, Initial, Expected, Operand, Vshort, Kokkos::atomic_fetch_add);
  testAtomic2(atomic_fetch_add, unsigned int, Initial, Expected, Operand, Vuint, Kokkos::atomic_fetch_add);
  testAtomic2(atomic_fetch_add, int, Initial, Expected, Operand, Vint, Kokkos::atomic_fetch_add);
  testAtomic2(atomic_fetch_add, unsigned long long int, Initial, Expected, Operand, Vull, Kokkos::atomic_fetch_add);

  Expected=0;
  Initial=M;
  Operand = 1;
  std::cout << "\n\nChecking atomic_fetch_sub: " << std::endl;
  //testAtomic2(atomic_fetch_sub, char, Initial, Expected, Operand, Vchar, Kokkos::atomic_fetch_sub);
  //testAtomic2(atomic_fetch_sub, short, Initial, Expected, Operand, Vshort, Kokkos::atomic_fetch_sub);
  testAtomic2(atomic_fetch_sub, unsigned int, Initial, Expected, Operand, Vuint, Kokkos::atomic_fetch_sub);
  testAtomic2(atomic_fetch_sub, int, Initial, Expected, Operand, Vint, Kokkos::atomic_fetch_sub);

  std::cout << "\n\nChecking atomic_fetch_min: " << std::endl;
  testAtomic2(atomic_fetch_min, unsigned int, m, 0, m, Vuint, Kokkos::atomic_fetch_min);
  testAtomic2(atomic_fetch_min, int, m, -(M-1), -m, Vint, Kokkos::atomic_fetch_min);
  testAtomic2(atomic_fetch_min, unsigned long long int, m, 0, m, Vull, Kokkos::atomic_fetch_min);
  //[ERROR] CUDA intrinsic: Not supported, OpenACC atomic: Not implemented, Serial template: Not implemented
  //testAtomic2(atomic_fetch_min, long long int, m, -(M-1), -m, Vll, Kokkos::atomic_fetch_min);

  std::cout << "\n\nChecking atomic_fetch_max: " << std::endl;
  testAtomic2(atomic_fetch_max, unsigned int, 0, M-1, m, Vuint, Kokkos::atomic_fetch_max);
  //[DEBUG] CUDA intrinsic: OK, OpenACC atomic: Not implemented, Serial template: Not implemented
  testAtomic2(atomic_fetch_max, int, 0, M-1, m, Vint, Kokkos::atomic_fetch_max);
  //[DEBUG] CUDA intrinsic: OK, OpenACC atomic: Not implemented, Serial template: Not implemented
  testAtomic2(atomic_fetch_max, unsigned long long int, 0, M-1, m, Vull, Kokkos::atomic_fetch_max);
  //[ERROR] CUDA intrinsic: Not supported, OpenACC atomic: Not implemented, Serial template: Not implemented
  //testAtomic2(atomic_fetch_max, long long int, 0, M-1, m, Vll, Kokkos::atomic_fetch_max);

  Expected = 0;
  for(int i=0; i<M; i++) Expected^=i;
  std::cout << "\n\nChecking atomic_fetch_xor: " << std::endl;
  testAtomic2(atomic_fetch_xor, unsigned int, 0, Expected, m, Vuint, Kokkos::atomic_fetch_xor);
  testAtomic2(atomic_fetch_xor, int, 0, Expected, m, Vint, Kokkos::atomic_fetch_xor);
  //[DEBUG] CUDA intrinsic: OK, OpenACC atomic: OK, Serial template: Unsupported local variable
  testAtomic2(atomic_fetch_xor, unsigned long long int, 0, Expected, m, Vull, Kokkos::atomic_fetch_xor);

  Expected = 0;
  for(int i=0; i<M; i++) Expected|=i;
  std::cout << "\n\nChecking atomic_fetch_or: " << std::endl;
  testAtomic2(atomic_fetch_or, unsigned int, 0, Expected, m, Vuint, Kokkos::atomic_fetch_or);
  testAtomic2(atomic_fetch_or, int, 0, Expected, m, Vint, Kokkos::atomic_fetch_or);
  //[ERROR] CUDA intrinsic: Unhandled builtin: 609 (__pgi_atomicOrul), OpenACC atomic: wrong result, Serial template: Unsupported local variable
  //testAtomic2(atomic_fetch_or, unsigned long long int, 0, Expected, m, Vull, Kokkos::atomic_fetch_or);

  Expected = 0;
  for(int i=0; i<M; i++) Expected&=i;
  std::cout << "\n\nChecking atomic_fetch_and: " << std::endl;
  testAtomic2(atomic_fetch_and, unsigned int, 0xFFFFFFFF, Expected, m, Vuint, Kokkos::atomic_fetch_and);
  testAtomic2(atomic_fetch_and, int, 0xFFFFFFFF, Expected, m, Vint, Kokkos::atomic_fetch_and);
  //[ERROR] CUDA intrinsic: Unhandled builtin: 605 (__pgi_atomicAndul), OpenACC atomic: wrong result, Serial template: Unsupported local variable
  //testAtomic2(atomic_fetch_and, unsigned long long int, 0xFFFFFFFFFFFFFFFF, Expected, m, Vull, Kokkos::atomic_fetch_and);

  std::cout << "\n\nFree Kokkos data: " << std::endl;
  Kokkos::kokkos_free<>(Vchar);
  Kokkos::kokkos_free<>(Vshort);
  Kokkos::kokkos_free<>(Vfloat);
  Kokkos::kokkos_free<>(Vuint);
  Kokkos::kokkos_free<>(Vint);
  Kokkos::kokkos_free<>(Vull);
  Kokkos::kokkos_free<>(Vll);

  }
  
  Kokkos::finalize();

  return 0;
}
