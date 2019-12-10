#ifndef cudavectors_h
#define cudavectors_h

namespace cudavectors {

  struct CylindricalVector {
    float rho;
    float eta;
    float phi;
  };

  struct CartesianVector {
    float x;
    float y;
    float z;
  };

  void convertWrapper(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size);

}  // namespace cudavectors

#endif  // cudavectors_h
