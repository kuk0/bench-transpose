#include <algorithm>
#include <benchmark/benchmark.h>
#include <vector>

#define MAX 26000
#define PAD 47
#define ARGS CustomArguments
#define START 1000

using namespace std;

// we store the matrix as flat 1D array
// instead of vector<vector<int>> - this performs slightly better
typedef vector<int> vi;

// adds padding to N;
// returns N' > N such that N'%64 == PAD
// due to small cache associativity the "row-by-row" algorithm
// performs poorly for N divisible by high power of 2:
// related elements evict each other from cache
int pad(int N) {
  return N + (PAD ? (64+PAD-N%64)%64 : 0);
}

// transpose a matrix row by row
static void BM_Transpose_Row(benchmark::State& state) {
  int N = state.range(0);
  int N2 = pad(N);
  vi m(N2*N, 1);
  for (auto _ : state) {
    for (int i=0; i<N; ++i) {
      for (int j=i+1; j<N; ++j) {
        swap(m[i*N2+j], m[j*N2+i]);
      }
    }
  }
}

// transpose a matrix: split it into blocks B*B
// and transpose&move each block
// this is more cache-friendly than simple row-by-row
static void BM_Transpose_Block(benchmark::State& state) {
  int N = state.range(0);
  int N2 = pad(N);
  int B = 64; // works well on my computer
  vi m(N2*N, 1);
  for (auto _ : state) {
    for (int k=0; k<N; k+=B) {
      // diagonal
      for (int i=k; i<k+B && i<N; ++i) {
        for (int j=k+1; j<k+B && j<N; ++j) {
          swap(m[i*N2+j], m[j*N2+i]);
        }
      }
      // off-diagonal
      for (int l=k+B; l<N; l+=B) {
        for (int i=k; i<k+B && i<N; ++i) {
          for (int j=l; j<l+B && j<N; ++j) {
            swap(m[i*N2+j], m[j*N2+i]);
          }
        }
      }
    }
  }
}

// transpose a matrix: split it into blocks B2*B2
// then split those blocks further into B*B blocks
// and transpose&move each block
// this is supposed to be even more cache-friendly
// than row-by-row and block-by-block
static void BM_Transpose_Block2(benchmark::State& state) {
  int N = state.range(0);
  int N2 = pad(N);
  int B = 4;
  int B2 = 1040;
  vi m(N2*N, 1);
  for (auto _ : state) {
    for (int x=0; x<N; x+=B2) {
      for (int k=x; k<x+B2 && k<N; k+=B) {
        for (int i=k; i<k+B && i<N; ++i) {
          for (int j=k+1; j<k+B && j<N; ++j) {
            swap(m[i*N2+j], m[j*N2+i]);
          }
        } 
        for (int l=k+B; l<x+B2 && l<N; l+=B) {
          for (int i=k; i<k+B && i<N; ++i) {
            for (int j=l; j<l+B && j<N; ++j) {
              swap(m[i*N2+j], m[j*N2+i]);
            }
          }
        }
      }
      for (int y=x+B2; y<N; y+=B2) {
        for (int k=x; k<x+B2 && k<N; k+=B) {
          for (int l=y; l<y+B2 && l<N; l+=B) {
            for (int i=k; i<k+B && i<N; ++i) {
              for (int j=l; j<l+B && j<N; ++j) {
                swap(m[i*N2+j], m[j*N2+i]);
              }
            }
          }
        }
      }
    }
  }
}

// transpose a matrix recursively:
// (A B)'  ->   (A' C')
// (C D)   ->   (B' D')
// note: we need to tranpose+swap B and C
void transpose_rec(const int N, int n, vi &m,
                   int i0, int j0, int i1, int j1) {
  // transpose and swap submatrices of size n
  // with top-left corners (i0,j0) and (i1,j1)
  if (n <= 4) {
    if (i0==j0) {
      for (int i=i0; i<i0+n && i<N; ++i) {
        for (int j=i+1; j<i0+n && j<N; ++j) {
          swap(m[i*N+j], m[j*N+i]);
        }
      }
    } else {
      for (int i=i0, jj=j1; i<i0+n && i<N && jj<N; ++i, ++jj) {
        for (int j=j0, ii=i1; j<j0+n && j<N && ii<N; ++j, ++ii) {
          swap(m[i*N+j], m[ii*N+jj]);
        }
      }
    }
  } else {
    int h = n/2;
    transpose_rec(N, h, m, i0, j0, i1, j1);
    transpose_rec(N, n-h, m, i0+h, j0+h, i1+h, j1+h);
    transpose_rec(N, n-h, m, i0+h, j0, i1, j1+h);
  }
}

static void BM_Transpose_Rec(benchmark::State& state) {
  int N = state.range(0);
  vi m(N*N, 1);
  for (auto _ : state) {
    transpose_rec(N, N, m, 0, 0, 0, 0);
  }
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int i = START; i <= MAX; i = (i*12)/11) {
    b->Args({i});
  }
}
static void CustomArguments2(benchmark::internal::Benchmark* b) {
  for (int i = 64; i <= 4096; i += 64) {
    b->Args({i});
  }
}

static void BlockArgs(benchmark::internal::Benchmark* b) {
  for (int B2=4; B2 <= 80; B2 += 4) {
    b->Args({4096, B2});
  }
}

/*
BENCHMARK(BM_Transpose_Row)->RangeMultiplier(4)->Range(64,MAX);
BENCHMARK(BM_Transpose_Block)->RangeMultiplier(4)->Range(64,MAX);
BENCHMARK(BM_Transpose_Block2)->RangeMultiplier(4)->Range(64,MAX);
BENCHMARK(BM_Transpose_Rec)->RangeMultiplier(4)->Range(64,MAX);
*/

//BENCHMARK(BM_Transpose_Block)->Apply(BlockArgs);

BENCHMARK(BM_Transpose_Row)->Apply(ARGS);
BENCHMARK(BM_Transpose_Block)->Apply(ARGS);
BENCHMARK(BM_Transpose_Block2)->Apply(ARGS);
BENCHMARK(BM_Transpose_Rec)->Apply(ARGS);

BENCHMARK_MAIN();
