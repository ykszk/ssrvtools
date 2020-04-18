#include <thread>
#include <vector>
#include <algorithm>
using namespace std;

#define PROD(v) v[0]*v[1]*v[2]

typedef unsigned char T;
void modefilt3(const T *input, T *output, const unsigned int *dims, int size, int n_threads)
{
  auto max_value = *std::max_element(input, input + PROD(dims));
  std::vector<std::thread> threads(n_threads);
  for (int tid = 0; tid < n_threads; ++tid) {
    auto f = [=]() {
      std::vector<short> hist(max_value + 1);
      for (int z = 0; z < dims[2]; ++z) {
        for (int y = 0; y < dims[1]; ++y) {
          for (int x = tid; x < dims[0]; x += n_threads) {
            std::fill(hist.begin(), hist.end(), 0);
            for (int k = -size / 2; k <= size / 2; ++k) {
              int zz = std::min(std::max(z + k, 0), static_cast<int>(dims[2]) - 1);
              for (int j = -size / 2; j <= size / 2; ++j) {
                int yy = std::min(std::max(y + j, 0), static_cast<int>(dims[1]) - 1);
                for (int i = -size / 2; i <= size / 2; ++i) {
                  int xx = std::min(std::max(x + i, 0), static_cast<int>(dims[0]) - 1);
                  ++hist[input[zz*dims[1] * dims[0] + yy*dims[0] + xx]];
                }
              }
            }
            output[z*dims[0]*dims[1] + y*dims[0] + x] = std::distance(hist.begin(), std::max_element(hist.begin(), hist.end()));
          }
        }
      }
    };
    threads[tid] = std::thread(f);
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

typedef unsigned char PixelType;

py::array_t<PixelType, py::array::c_style> mode_filter(py::array_t<PixelType, py::array::c_style> arr, int filter_size, int concurrency)
{
  if (arr.ndim() != 3) {
    throw(std::invalid_argument("Invalid ndim. 3 is expected."));
  }
  if (concurrency <= 0) {
    concurrency = std::thread::hardware_concurrency();
  }
  auto shape = arr.shape();
  py::array::ShapeContainer sc(shape, shape + 3);
  py::array_t<PixelType, py::array::c_style> result(sc);
  unsigned int dims[] = { static_cast<unsigned int>(shape[2]), static_cast<unsigned int>(shape[1]), static_cast<unsigned int>(shape[0]) };
  modefilt3(static_cast<const PixelType*>(arr.data()), static_cast<PixelType*>(result.mutable_data()), dims, filter_size, concurrency);
  return result;
}


PYBIND11_MODULE(_label_filters, m)
{
  m.doc() = "Mode filter.";
  m.def("modefilt3",&mode_filter,"3D mode filter.");
}
