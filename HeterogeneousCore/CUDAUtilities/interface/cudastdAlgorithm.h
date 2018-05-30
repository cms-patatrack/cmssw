#ifndef HeterogeneousCoreCUDAUtilitiesstdalgorithm_h
#define HeterogeneousCoreCUDAUtilitiesstdalgorithm_h

#include <utility>

// reference implementation of std algorithm able to compile with cuda and run on  gpus
// mostly is adding a constexpr
namespace cuda_std  {

  template< typename T = void >
  struct less {
    constexpr bool operator()(const T &lhs, const T &rhs) const {
      return lhs < rhs;
    }
  };

  template<>
  struct less<void> {
    template< class T, class U>
    constexpr bool operator()(const T &lhs, const U &rhs ) const { return lhs < rhs;}
  };

  template<class RandomIt, class T, class Compare=less<T>>
  constexpr
  RandomIt lower_bound(RandomIt first, RandomIt last, const T& value, Compare comp={})
  {
    auto count = last-first;
 
    while (count > 0) {
        auto it = first;
        auto step = count / 2;
        it+=step;
        if (comp(*it, value)) {
            first = ++it;
            count -= step + 1;
        }
        else {
            count = step;
        }
    }
    return first;
  }


  template<class RandomIt, class T, class Compare=less<T>>
  constexpr
  RandomIt upper_bound(RandomIt first, RandomIt last, const T& value, Compare comp={})
  {
    auto count = last-first;
 
    while (count > 0) {
        auto it = first; 
        auto step = count / 2; 
        it+=step;
        if (!comp(value,*it)) {
            first = ++it;
            count -= step + 1;
        } 
        else {
            count = step;
        }
    }
    return first;
  }


  template<class RandomIt, class T, class Compare=cuda_std::less<T>>
  constexpr
  RandomIt binary_find(RandomIt first, RandomIt last, const T& value, Compare comp={})
  {
    first = cuda_std::lower_bound(first, last, value, comp);
    return first != last && !comp(value, *first) ? first : last;
  }


}


#endif
